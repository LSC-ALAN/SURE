from src.config.default import _CN as cfg

cfg.TRAINER.CANONICAL_BS = 8 * 14
cfg.TRAINER.CANONICAL_LR = 1e-3
cfg.TRAINER.WARMUP_STEP = int(304000 / cfg.TRAINER.CANONICAL_BS * 1)  # 1 epochs
cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
cfg.TRAINER.EPI_ERR_THR = 5e-4

cfg.SURE.COARSE.MCONF_THR = 0.2
cfg.SURE.FINE.SIGMA_THR = 1e-6
cfg.SURE.COARSE.BORDER_RM = 0

# Top-K should not exceed grid_size = TEST_RES_H / 8 * TEST_RES_W / 8 # 4800
# The recommended value is approximately grid_size * 0.35 for ScanNet
cfg.SURE.COARSE.TOPK = int(480 / 8 * 640 / 8 * 0.35)  # 1680
