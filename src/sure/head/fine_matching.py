import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import time

def compute_uncertainty(la, alpha, beta):
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (alpha - 1) / la
    return aleatoric, epistemic

import torch.nn as nn
import torch

# Depthwise Separable Conv1d + BN + Activation
class DWConv1d_BN_Act(nn.Sequential):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1, act=None, drop=None):
        super().__init__()
        self.add_module("dw", nn.Conv1d(in_ch, in_ch, ks, stride, pad, groups=in_ch, bias=False))
        self.add_module("pw", nn.Conv1d(in_ch, out_ch, 1, bias=False))
        self.add_module("bn", nn.BatchNorm1d(out_ch))
        if act: self.add_module("act", act)
        if drop: self.add_module("drop", nn.Dropout(drop))

# LayerNorm + Linear 模拟 Conv1d(ks=1)
class LN_Linear(nn.Sequential):
    def __init__(self, in_ch, out_ch, act=None, drop=None):
        super().__init__()
        self.add_module("ln", nn.LayerNorm(in_ch))
        self.add_module("linear", nn.Linear(in_ch, out_ch))
        if act: self.add_module("act", act)
        if drop: self.add_module("drop", nn.Dropout(drop))

    def forward(self, x):  # [B, C, L] -> [B, L, C] -> [B, L, C_out] -> [B, C_out, L]
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)

class TokenWiseMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, drop=0.0, act=nn.ReLU()):
        super().__init__()
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act,
            nn.Dropout(drop),
            # nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        x = x.view(B * N, C)
        x = self.net(x)
        x = x.view(B, N, -1)
        return x


class Conv1d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv1d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm1d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))


def soft_argmax(x, temperature=1.0):
    L = x.shape[1]
    assert L % 2  # L is odd to ensure symmetry
    idx = torch.arange(0, L, 1, device=x.device).repeat(x.shape[0], 1)
    scale_x = x / temperature
    out = F.softmax(scale_x, dim=1) * idx
    out = out.sum(dim=1, keepdim=True)

    return out


class FineMatching(nn.Module):
    def __init__(self, config, act_layer=nn.GELU):
        super(FineMatching, self).__init__()
        self.config = config

        self.block_dims = self.config["backbone"]["block_dims"]
        self.local_resolution = self.config["local_resolution"]
        self.drop = self.config["fine"]["droprate"]
        self.coord_length = self.config["fine"]["coord_length"]
        self.bi_directional_refine = self.config["fine"]["bi_directional_refine"]
        self.sigma_selection = self.config["fine"]["sigma_selection"]
        self.mconf_thr = self.config["coarse"]["mconf_thr"]
        self.sigma_thr = self.config["fine"]["sigma_thr"]
        self.border_rm = self.config["coarse"]["border_rm"] * \
            self.local_resolution
        self.deploy = self.config["deploy"]

        # network
        self.query_encoder = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
        )

        self.reference_encoder = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
        )

        self.merge_qr = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
        )

        self.x_head = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
            nn.Conv1d(self.block_dims[-1] * 2,
                      self.coord_length + 4, kernel_size=1),
        )

        self.y_head = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
            nn.Conv1d(self.block_dims[-1] * 2,
                      self.coord_length + 4, kernel_size=1),

        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def get_uncertainty(self, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return v, alpha, beta

    def compute_uncertainty(self, la, alpha, beta):
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (alpha - 1) / la
        return aleatoric, epistemic

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data={}):
    # def forward(self, feat_c0, feat_c1, data={}):

        t=time.time()
        q = self.query_encoder(
            feat_f0.permute(0, 2, 1).contiguous()
            + feat_c0.permute(0, 2, 1).contiguous()
        )
        r = self.reference_encoder(
            feat_f1.permute(0, 2, 1).contiguous()
            + feat_c1.permute(0, 2, 1).contiguous()
        )
        out = self.merge_qr(torch.cat([q, r], dim=1))

        x = self.x_head(out).permute(0, 2, 1).contiguous()
        y = self.y_head(out).permute(0, 2, 1).contiguous()

        
        if self.bi_directional_refine:
            x01, x10 = x.chunk(2, dim=1)
            x01 = x01.reshape(-1, self.coord_length + 4)
            x10 = x10.reshape(-1, self.coord_length + 4)
            x_out = torch.cat([x01, x10])

            y01, y10 = y.chunk(2, dim=1)
            y01 = y01.reshape(-1, self.coord_length + 4)
            y10 = y10.reshape(-1, self.coord_length + 4)
            y_out = torch.cat([y01, y10])
        else:
            x_out = x.reshape(-1, self.coord_length + 4)
            y_out = y.reshape(-1, self.coord_length + 4)

        x_cls = x_out[:, : self.coord_length + 1]
        coord_x = soft_argmax(x_cls) / self.coord_length - \
            0.5  # range [-0.5, +0.5]

        y_cls = y_out[:, : self.coord_length + 1]
        coord_y = soft_argmax(y_cls) / self.coord_length - 0.5
        coord = torch.cat([coord_x, coord_y], dim=1)

        loglax, logalphax, logbetax = x_out[:, -3:].chunk(3, dim=1)
        loglay, logalphay, logbetay = y_out[:, -3:].chunk(3, dim=1)

        # [N, 1] + [N, 1] => [N, 2]
        logla = torch.cat([loglax, loglay], dim=1)
        logalpha = torch.cat([logalphax, logalphay], dim=1)
        logbeta = torch.cat([logbetax, logbetay], dim=1)

        v, alpha, beta=self.get_uncertainty(logla,logalpha,logbeta)
        aleatoric, epistemic = self.compute_uncertainty(v, alpha, beta)
        aleatoric_mean = aleatoric.mean(dim=1)
        epistemic_mean = epistemic.mean(dim=1)
        data.update({
            "aleatoric_mean":aleatoric_mean,
            "epistemic_mean":epistemic_mean,
            "coord":coord
        })
        if data.get("target_uv", None) is not None:
            self.compute_corr(data)

            gt_uv = data["target_uv"]
            mask = data["target_uv_weight"].clone()

            if mask.sum() == 0:
                mask[0] = True
            mask_coord = coord[mask]
            mask_gt_uv = gt_uv[mask]

            v, alpha, beta=v[mask], alpha[mask], beta[mask]

            data.update(
                {
                    "pred_coord": coord,
                    "mask_coord": mask_coord,
                    "v": v,
                    "alpha":alpha,
                    "beta":beta
                }
            )
        else:
            data.update(
                {
                    "pred_coord": coord,
                }
            )

        if not self.deploy:
            self.final_matching_selection(data)


    def compute_corr(self, data,):
        from scipy.stats import spearmanr

        gt_uv = data["target_uv"]  # [M, 2]
        pred_uv = data["coord"]  # [N, 2]
        epe = torch.norm(pred_uv - gt_uv, dim=1)  # [N]

        aleatoric=data["aleatoric_mean"]
        epistemic=data["epistemic_mean"]

        def pearson_corr(x, y):
            vx = x - x.mean()
            vy = y - y.mean()
            return (vx * vy).mean() / (x.std() * y.std())

        def spearman_corr(x: torch.Tensor, y: torch.Tensor):
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            corr, _ = spearmanr(x_np, y_np)
            return torch.tensor(corr, device=x.device)

        corr_aleatoric = spearman_corr(epe, aleatoric)
        corr_epistemic = spearman_corr(epe, epistemic)
        corr_alea_epi = spearman_corr(aleatoric, epistemic)

        data.update({"corr_aleatoric": corr_aleatoric.cpu().item(), "corr_epistemic": corr_epistemic.cpu().item(),
                     "corr_alea_epi": corr_alea_epi.cpu().item()})
        print(corr_aleatoric,corr_epistemic,corr_alea_epi)

    @torch.no_grad()
    def final_matching_selection(self, data):
        offset = data["pred_coord"] * self.local_resolution

        if self.bi_directional_refine:
            fine_offset01, fine_offset10 = torch.clamp(
                offset, -self.local_resolution / 2, self.local_resolution / 2
            ).chunk(2)
        else:
            fine_offset01 = torch.clamp(
                offset, -self.local_resolution / 2, self.local_resolution / 2
            )

        h0, w0 = data["hw0_i"]
        h1, w1 = data["hw1_i"]
        scale0 = data["scale0"][data["b_ids"]] if "scale0" in data else 1.0
        scale1 = data["scale1"][data["b_ids"]] if "scale1" in data else 1.0
        scale0_w = scale0[:, 0] if "scale0" in data else 1.0
        scale0_h = scale0[:, 1] if "scale0" in data else 1.0
        scale1_w = scale1[:, 0] if "scale1" in data else 1.0
        scale1_h = scale1[:, 1] if "scale1" in data else 1.0

        # Filter by mconf and border
        mkpts0_f = data["mkpts0_c"]
        mkpts1_f = data["mkpts1_c"] + fine_offset01 * scale1

        mask = (
            (data["mconf"] > self.mconf_thr)

            & (mkpts0_f[:, 0] >= self.border_rm)
            & (mkpts0_f[:, 0] <= w0 * scale0_w - self.border_rm)
            & (mkpts0_f[:, 1] >= self.border_rm)
            & (mkpts0_f[:, 1] <= h0 * scale0_h - self.border_rm)
            & (mkpts1_f[:, 0] >= self.border_rm)
            & (mkpts1_f[:, 0] <= w1 * scale1_w - self.border_rm)
            & (mkpts1_f[:, 1] >= self.border_rm)
            & (mkpts1_f[:, 1] <= h1 * scale1_h - self.border_rm)
        )
        if self.bi_directional_refine:
            mkpts0_f_ = data["mkpts0_c"] + fine_offset10 * scale0

            mkpts1_f_ = data["mkpts1_c"]
            mask_ = (
                (data["mconf"] > self.mconf_thr)
                & (mkpts0_f_[:, 0] >= self.border_rm)
                & (mkpts0_f_[:, 0] <= w0 * scale0_w - self.border_rm)
                & (mkpts0_f_[:, 1] >= self.border_rm)
                & (mkpts0_f_[:, 1] <= h0 * scale0_h - self.border_rm)
                & (mkpts1_f_[:, 0] >= self.border_rm)
                & (mkpts1_f_[:, 0] <= w1 * scale1_w - self.border_rm)
                & (mkpts1_f_[:, 1] >= self.border_rm)
                & (mkpts1_f_[:, 1] <= h1 * scale1_h - self.border_rm)
            )

        if self.bi_directional_refine:
            mkpts0_f = torch.cat([mkpts0_f, mkpts0_f_])
            mkpts1_f = torch.cat([mkpts1_f, mkpts1_f_])

            aleatoric_threshold = data["aleatoric_mean"].quantile(0.95)
            epistemic_threshold = data["epistemic_mean"].quantile(0.95)
            mask = torch.cat([mask, mask_]) & (data["aleatoric_mean"] < aleatoric_threshold)& (data["epistemic_mean"] < epistemic_threshold)


            data["mconf"] = torch.cat([data["mconf"], data["mconf"]])
            data["b_ids"] = torch.cat([data["b_ids"], data["b_ids"]])

        data.update(
            {
                "m_bids": data["b_ids"][mask],
                "mkpts0_f": mkpts0_f[mask],
                "mkpts1_f": mkpts1_f[mask],
                "mconf": data["mconf"][mask],
                "aleatoric_mean":data["aleatoric_mean"][mask],
                "epistemic_mean":data["epistemic_mean"][mask]
            }
        )
