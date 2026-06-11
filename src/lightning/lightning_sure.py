from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import lightning.pytorch as pl
from matplotlib import pyplot as plt

from src.sure import SURE
from src.sure.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
import os
from src.losses.sure_loss import SURELoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_SURE(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)

        self.profiler = profiler or PassThroughProfiler()

        self.n_vals_plot = 70
        # self.n_vals_plot = max(
        #     config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        # )

        # Matcher: SURE
        self.matcher = SURE(config=_config["sure"])
        self.loss = SURELoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")[
                "state_dict"]
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.dump_dir = dump_dir

        # outputs
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("SURE"):
            with torch.autocast(enabled=self.config.SURE.MP, device_type="cuda"):
                self.matcher(batch)

        # with self.profiler.profile("Compute fine supervision"):
        #     with torch.autocast(enabled=False, device_type='cuda'):
        #         compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.SURE.MP, device_type="cuda"):
                self.loss(batch)

    def _compute_metrics(self, batch):
        # compute epi_errs for each match
        compute_symmetrical_epipolar_errors(batch)

        compute_pose_errors(
            batch, self.config
        )  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
            "epi_errs": [
                (batch["epi_errs"].reshape(-1, 1))[batch["m_bids"] == b]
                .reshape(-1)
                .cpu()
                .numpy()
                for b in range(bs)
            ],
            "R_errs": batch["R_errs"],
            "t_errs": batch["t_errs"],
            "inliers": batch["inliers"],
            "num_matches": [batch["mconf"].shape[0]],  # batch size = 1 only
            "corr_epistemic":[batch["corr_epistemic"]],
            "corr_aleatoric": [batch["corr_aleatoric"]],
            "corr_alea_epi":[batch["corr_alea_epi"]]
        }
        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(
                    f"train/{k}", v, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(
                    batch
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

        out = {"loss": batch["loss"]}
        self.log("loss", batch["loss"], prog_bar=True, rank_zero_only=True)

        # avoid significant memory growth
        # self.train_step_outputs.append(out)
        return out

    def on_after_backward(self) -> None:
        for n, p in self.named_parameters():
            if p.grad is None:
                print(n)
        return super().on_after_backward()

    def on_train_epoch_end(self):
        pass  # avoid significant memory growth during training
        # outputs = self.train_step_outputs
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # if self.trainer.global_rank == 0:
        #     self.logger.experiment.add_scalar(
        #         'train/avg_loss_on_epoch', avg_loss,
        #         global_step=self.current_epoch)
        # self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(
            self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        out = {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        # handle multiple validation sets
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if self.trainer.ckpt_path is None and self.trainer.sanity_checking:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(
                    flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(
                    val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(
                    gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                                fig,
                                cur_epoch,
                                close=True,
                            )


            plt.close("all")

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(
                f"auc@{thr}",
                torch.tensor(np.mean(multi_val_metrics[f"auc@{thr}"])),
                sync_dist=True,
            )  # ckpt monitors on this
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # with self.profiler.profile("Compute coarse supervision"):
        #     with torch.autocast(enabled=False, device_type="cuda"):
        #         compute_supervision_coarse(batch, self.config)
        compute_supervision_coarse(batch, self.config)

        if self.config.SURE.HALF:
            self.matcher = self.matcher.eval().half()

        # Following EfficientLoFTR
        if not self.warmup:
            if self.config.SURE.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.SURE.MP, device_type="cuda"):
                    for i in range(50):
                        self.matcher(batch)
                        self.loss(batch)
            self.warmup = True

        torch.cuda.synchronize()
        if self.config.SURE.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.SURE.MP, device_type="cuda"):
                self.start_event.record()
                self.matcher(batch)
                self.loss(batch)

                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        if self.dump_dir is not None:
            with self.profiler.profile("dump_results"):
                # dump results for further analysis
                keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ["R_errs", "t_errs", "inliers"]:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        self.test_step_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]

        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # dump
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"]
                                 for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        # [{key: [{...}, *#bs]}, *#batch]
        if self.trainer.global_rank == 0:
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )

            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            print(
                "Averaged Matching time over 1500 pairs: {:.2f} ms".format(
                    self.total_ms / 1500
                )
            )
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "SURE_pred_eval", dumps)

        self.test_step_outputs.clear()
