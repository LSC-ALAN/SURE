import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_sure import PL_SURE

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_trainval_832.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/sure/outdoor/sure_base.py")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--batch_size", type=int,
                        default=2, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint path, helpful for using a pre-trained coarse-only SURE",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--num_sanity_val_steps", type=int, default=10)
    parser.add_argument("--benchmark", default=True)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--split_data_idx", type=int, default=0)
    parser.add_argument(
        "--parallel_load_data",
        default=False,
        action="store_true",
        help="load datasets in with multiple processes.",
    )
    parser.add_argument(
        "--disable_ckpt",
        action="store_true",
        help="disable checkpoint saving (useful for debugging).",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_SURE(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"SURE LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"SURE DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(
        monitor="auc@10",
        verbose=True,
        save_top_k=10,
        mode="max",
        save_last=True,
        save_weights_only=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp",
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        # limit_val_batches=1,
        num_sanity_val_steps=args.num_sanity_val_steps,
        benchmark=True,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        use_distributed_sampler=False,
        profiler=profiler,
        # weights_summary='full'
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    # print(model)
    def print_model(module, indent=0):
        for name, child in module.named_children():
            num_params = sum(p.numel() for p in child.parameters())
            # 计算内存占用，float32 每个参数4字节
            mem_mb = num_params * 4 / (1024 ** 2)
            print("  " * indent + f"{name}: {child.__class__.__name__}, params: {num_params}, mem: {mem_mb:.3f} MB")
            print_model(child, indent + 1)

    print_model(model)

    trainer.fit(model, datamodule=data_module, ckpt_path=None)
if __name__ == "__main__":
    main()
