from src.config.default import _CN as cfg

cfg.TRAINER.CANONICAL_BS = 4 * 8
cfg.TRAINER.CANONICAL_LR = 2e-3
cfg.TRAINER.WARMUP_STEP = int(36800 / cfg.TRAINER.CANONICAL_BS * 3)  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.TRAINER.EPI_ERR_THR = 1e-4

cfg.SURE.COARSE.MCONF_THR = 0.05
cfg.SURE.FINE.SIGMA_THR = 1e-6
cfg.SURE.COARSE.BORDER_RM = 0

# Top-K should not exceed grid_size = TEST_RES_H / 8 * TEST_RES_W / 8
# The recommended value is approximately grid_size * 0.35 for Megadepth
# cfg.SURE.COARSE.TOPK = int(832 / 8 * 832 / 8 * 0.35)  # 3786 for train & LO-RANSAC test
# cfg.SURE.COARSE.TOPK = int(480 / 8 * 640 / 8 * 0.35)  # 1680
# cfg.SURE.COARSE.TOPK = 1000  # 3786 for train & LO-RANSAC test

cfg.SURE.COARSE.TOPK = int(1152 / 8 * 1152 / 8 * 0.35)  # 7258 for test

