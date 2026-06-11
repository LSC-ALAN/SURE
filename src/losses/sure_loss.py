from loguru import logger

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class SURELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.coord_length = config["sure"]["fine"]["coord_length"]

        self.loss_config = config["sure"]["loss"]
        self.sparse_spvs = self.loss_config["sparse_spvs"]

        # coarse-level
        self.c_pos_w = self.loss_config["pos_weight"]
        self.c_neg_w = self.loss_config["neg_weight"]
        # fine-level
        self.q_distribution = self.loss_config["q_distribution"]
        # self.fine_type = self.loss_config["fine_type"]
        # self.fine_loss = [nn.L1Loss(), nn.MSELoss(), nn.SmoothL1Loss()][1]

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        if self.loss_config["coarse_type"] == "cross_entropy":
            assert (
                not self.sparse_spvs
            ), "Sparse Supervision for cross-entropy not implemented!"
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[pos_mask])
            loss_neg = -torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

        elif self.loss_config["coarse_type"] == "focal":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config["focal_alpha"]
            gamma = self.loss_config["focal_gamma"]

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = -alpha * \
                    torch.pow(1 - pos_conf, gamma) * pos_conf.log()

                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]

                loss = c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:
                loss_pos = (
                    -alpha
                    * torch.pow(1 - conf[pos_mask], gamma)
                    * (conf[pos_mask]).log()
                )

                loss_neg = (
                    -alpha
                    * torch.pow(conf[neg_mask], gamma)
                    * (1 - conf[neg_mask]).log()
                )
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]

                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError(
                "Unknown coarse loss: {type}".format(
                    type=self.loss_config["coarse_type"]
                )
            )

    import torch

    def kl_divergence_gaussian(self,gt, mu, sigma, p_sigma=0.1, eps=1e-9):
        """
        KL(P‖Q) where:
        - P: GT Gaussian(μ=gt, σ=p_sigma)
        - Q: predicted Gaussian(μ=mu, σ=sigma)
        """
        var_p = p_sigma ** 2
        var_q = sigma ** 2 + eps

        kl = (
                torch.log(sigma / p_sigma + eps)
                + (var_p + (gt - mu) ** 2) / (2 * var_q)
                - 0.5
        )
        return kl  # shape: [N, 2]

    def compute_rle_loss_kl(self,data, f_weight=1.0, p_sigma=0.1):
        """
        KL-based fine-level loss, for offset regression.
        """
        gt_uv = data["target_uv"]  # [M, 2]
        gt_uv_weight = data["target_uv_weight"]  # [M] or [M, 1]
        pred_uv = data["mask_coord"]  # [M, 2]
        pred_sigma = data["mask_sigma"]  # [M, 1]
        nf_loss = data.get("nf_loss", 0.0).mean()

        if gt_uv_weight.sum() == 0:
            if f_weight > 0 and hasattr(torch.nn.Module, 'training') and torch.nn.Module.training:
                print("Warning: No GT supervision, assigning dummy mask")
                gt_uv_weight[0] = 1
                f_weight = 0.0
            else:
                return None

        valid = gt_uv_weight.bool().squeeze()
        kl = self.kl_divergence_gaussian(
            gt=gt_uv[valid],
            mu=pred_uv,
            sigma=pred_sigma,
            p_sigma=p_sigma,
        )
        kl_loss = kl.mean()

        total_loss = f_weight * kl_loss + nf_loss
        return total_loss

    def logQ(self, gt_uv, pred_jts, sigma):
        assert self.q_distribution in ["laplace", "gaussian"]

        error = (pred_jts - gt_uv) / (sigma + 1e-9)

        if self.q_distribution == "laplace":
            loss_q = torch.log(sigma * 2) + torch.abs(error)
        else:
            loss_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

        return loss_q

    def compute_rle_loss(self, data, f_weight=1):
        gt_uv = data["target_uv"]
        gt_uv_weight = data["target_uv_weight"]

        if gt_uv_weight.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning(
                    "assign a false supervision to avoid ddp deadlock")
                gt_uv_weight[0] = True
                f_weight = 0.0
            else:
                return None

        residual = True
        if residual:
            Q_logprob = self.logQ(
                gt_uv[gt_uv_weight], data["mask_coord"], data["mask_sigma"]
            )
            loss = Q_logprob + data["nf_loss"]

        return loss.mean() * f_weight

    def generate_soft_target_1d(self,coord, num_bins, sigma):
        """
        input：
            coord: [N], [-0.5, 0.5]
            num_bins: class number，64
            sigma:
        output：
            soft target，shape [N, num_bins]
        """
        device = coord.device
        bins = torch.arange(num_bins, device=device).float()  # [num_bins]

        coord_bin = (coord + 0.5) * (num_bins - 1)  # [0, num_bins-1]
        coord_bin = coord_bin.unsqueeze(1)  # [N,1]

        dist = (bins.unsqueeze(0) - coord_bin) ** 2  # [N, num_bins]
        soft_target = torch.exp(-dist / (2 * sigma ** 2))
        soft_target /= soft_target.sum(dim=1, keepdim=True)  # 归一化

        return soft_target

    def kl_div_loss_1d(self,logits, soft_target):
        """
        logits: [N, C]
        soft_target: [N, C]
        """
        log_prob = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_prob, soft_target, reduction="batchmean")
        return loss

    def compute_simcc_kl_loss(self,data, num_bins=17, sigma=1.5, f_weight=1.0):
        """
        data contain：
            pred_x_logits: [N, num_bins]
            pred_y_logits: [N, num_bins]
            target_uv: [N, 2] [-0.5, 0.5]
            target_uv_weight: [N] bool mask
        """
        target_uv = data["target_uv"]
        target_uv_weight = data["target_uv_weight"].bool()
        pred_x_logits = data["pred_x_logits"]
        pred_y_logits = data["pred_y_logits"]
        # nf_loss = data.get("nf_loss", 0.0).mean()

        valid = target_uv_weight
        if target_uv_weight.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                target_uv_weight[0] = True
                f_weight = 0.0
            else:
                return None

        target_uv = target_uv[valid]
        pred_x_logits = pred_x_logits[valid]
        pred_y_logits = pred_y_logits[valid]

        target_x_soft = self.generate_soft_target_1d(target_uv[:, 0], num_bins, sigma)
        target_y_soft = self.generate_soft_target_1d(target_uv[:, 1], num_bins, sigma)

        loss_x = self.kl_div_loss_1d(pred_x_logits, target_x_soft)
        loss_y = self.kl_div_loss_1d(pred_y_logits, target_y_soft)

        return (loss_x + loss_y) * f_weight

    #     return self.fine_loss(gt_uv[gt_uv_weight], pred_jts[gt_uv_weight]) * f_weight
    def compute_sar_loss(self, data, f_weight=1.0, lambda_reg=10.0):
        """
        Spatial-Aware Regression (SAR) loss function.
        Assumes:
            - pred_uv  delta offset,  [-0.5, 0.5]
            - conf_logits [N, 2]，x/y direction confidence logits
        Args:
            data: dict contain target_uv, target_uv_weight, mask_coord, mask_sigma, nf_loss
            f_weight: SAR loss weight
            lambda_reg: control exp(-λ * error)  sharpness
        Returns:
            scalar loss
        """
        gt_uv = data["target_uv"]  # [M, 2]
        gt_uv_weight = data["target_uv_weight"]  # [M] or [M, 1]
        pred_uv = data["mask_coord"]  # [N, 2]，
        conf_logits = data["mask_sigma"]  # [N, 2]， confidence logits
        nf_loss = data.get("nf_loss", 0.0).mean()

        if gt_uv_weight.sum() == 0:
            if f_weight > 0 and self.training:
                print("Warning: No GT supervision, assigning dummy mask")
                gt_uv_weight[0] = 1
                f_weight = 0.0
            else:
                return None

        valid = gt_uv_weight.bool().squeeze()
        gt_uv = gt_uv[valid]  # [N, 2]
        conf_logits = conf_logits  # [N, 2]

        # 1. Regression score r_t = exp(-λ * L1(pred - gt))
        error = torch.abs(pred_uv - gt_uv)  # [N, 2]
        r_t = torch.exp(-lambda_reg * error)  # [N, 2]

        # 2. Confidence score（soft selector）
        c_t = conf_logits  # [N, 2]
        w_t = c_t / (c_t.sum(dim=0, keepdim=True) + 1e-8)  # [N, 2]

        # 3. SAR loss = -log(Σ w_t * r_t)
        sar_loss_xy = -torch.log((w_t * r_t).sum(dim=0) + 1e-8)  # [2]
        sar_loss = sar_loss_xy.mean()

        return f_weight * sar_loss + nf_loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        """compute element-wise weights for computing coarse-level loss."""
        if "mask0" in data:
            c_weight = (
                data["mask0"].flatten(-2)[..., None]
                * data["mask1"].flatten(-2)[:, None]
            ).float()
        else:
            c_weight = None
        return c_weight

    def criterion_uncertainty(self, data,f_weight=1.0):
        # om: 2 * beta * (1 + la)
        gt_uv = data["target_uv"]  # [M, 2]
        gt_uv_weight = data["target_uv_weight"]  # [M] or [M, 1]

        u = data["mask_coord"]  # [N, 2]
        y=gt_uv[gt_uv_weight]
        la =data["v"]
        alpha= data["alpha"]
        beta = data["beta"]

        om = 2 * beta * (1 + la)

        loss = torch.mean(
            0.5 * torch.log(np.pi / la)
            - alpha * torch.log(om)
            + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )

        lossr = torch.mean(
            torch.abs(u - y) * (2 * la + alpha)
        )

        return loss + lossr

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        # self.compute_corr(data)
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data["conf_matrix"],
            data["conf_matrix_gt"],
            weight=c_weight,
        )
        loss = loss_c * self.loss_config["coarse_weight"]
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        loss_f=self.criterion_uncertainty(
            data=data,
            f_weight=self.loss_config["fine_weight"],
        )
        if loss_f is not None:
            loss += loss_f
            loss_scalars.update(
                {"loss_f": min(loss_f.clone().detach().cpu(),
                               torch.tensor(1.0))}
            )
        else:
            assert self.training is False
            # 1 is the upper bound
            loss_scalars.update({"loss_f": torch.tensor(1.0)})

        loss_scalars.update({"loss": loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
