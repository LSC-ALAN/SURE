from configs.data.base import cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "xxx"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

cfg.SURE.TRAIN_RES_H = 832
cfg.SURE.TRAIN_RES_W = 832
cfg.SURE.TEST_RES_H = 1152
cfg.SURE.TEST_RES_W = 1152

cfg.SURE.NECK.NPE = [
    cfg.SURE.TRAIN_RES_H,
    cfg.SURE.TRAIN_RES_W,
    cfg.SURE.TEST_RES_H,
    cfg.SURE.TEST_RES_W,
]

