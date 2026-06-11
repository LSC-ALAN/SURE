from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  SURE Pipeline  ↓  ##############
_CN.SURE = CN()
_CN.SURE.TRAIN_RES_H = 832
_CN.SURE.TRAIN_RES_W = 832
_CN.SURE.TEST_RES_H = 832
_CN.SURE.TEST_RES_W = 832
_CN.SURE.LOCAL_RESOLUTION = 8  # coarse matching at 1/8, window_size 8x8 in fine_level
_CN.SURE.MP = False # Not use mixed precision (ELoFTR defualt used mixed precision)
_CN.SURE.HALF = False  # Not FP16
_CN.SURE.DEPLOY = False # export onnx model
_CN.SURE.EVAL_TIMES = 1

# 3. Coarse-Matching config
_CN.SURE.BACKBONE = CN()
_CN.SURE.BACKBONE.BLOCK_DIMS = [32, 64, 128, 256, 256]  # 1/2 -> 1/32



# 3. Coarse-Matching config
_CN.SURE.NECK = CN()
_CN.SURE.NECK.D_MODEL = 256
_CN.SURE.NECK.NHEAD = 8
_CN.SURE.NECK.LAYER_NAMES = ["self", "cross"] * 2
_CN.SURE.NECK.AGG_SIZE0 = 1
_CN.SURE.NECK.AGG_SIZE1 = 1
_CN.SURE.NECK.ROPE = True
_CN.SURE.NECK.NPE = None

# 3. Coarse-Matching config
_CN.SURE.COARSE = CN()
_CN.SURE.COARSE.MCONF_THR = 0.05
_CN.SURE.COARSE.BORDER_RM = 0
_CN.SURE.COARSE.DSMAX_TEMPERATURE = 0.1
_CN.SURE.COARSE.TRAIN_PAD_NUM = 32  # training tricks: avoid DDP deadlock
# _CN.SURE.COARSE.TOPK = 1000
_CN.SURE.COARSE.TOPK = int(480 / 8 * 640 / 8 * 0.35)  # 1680


_CN.SURE.COARSE.DS_OPT = True



_CN.SURE.COARSE.D_MODEL = 256
_CN.SURE.COARSE.D_FFN = 256
_CN.SURE.COARSE.NHEAD = 8
_CN.SURE.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.SURE.COARSE.AGG_SIZE0 = 4
_CN.SURE.COARSE.AGG_SIZE1 = 4
_CN.SURE.COARSE.NO_FLASH = False
_CN.SURE.COARSE.ROPE = True
_CN.SURE.COARSE.NPE = [832, 832, 832, 832] # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832



# 4. SURE-fine module config
_CN.SURE.FINE = CN()
_CN.SURE.FINE.DROPRATE = None
_CN.SURE.FINE.COORD_LENGTH = 16
_CN.SURE.FINE.BI_DIRECTIONAL_REFINE = True
_CN.SURE.FINE.SIGMA_THR = 0.0
_CN.SURE.FINE.SIGMA_SELECTION = True

# 5. SURE Losses
# -- # coarse-level
_CN.SURE.LOSS = CN()
_CN.SURE.LOSS.COARSE_TYPE = "focal"
_CN.SURE.LOSS.COARSE_WEIGHT = 1.0
_CN.SURE.LOSS.SPARSE_SPVS = True

# -- - -- # focal loss (coarse)
_CN.SURE.LOSS.FOCAL_ALPHA = 0.25
_CN.SURE.LOSS.FOCAL_GAMMA = 2.0
_CN.SURE.LOSS.POS_WEIGHT = 1.0
_CN.SURE.LOSS.NEG_WEIGHT = 1.0


# -- # fine-level
_CN.SURE.LOSS.FINE_TYPE = "rle"
_CN.SURE.LOSS.FINE_WEIGHT = 0.2
_CN.SURE.LOSS.Q_DISTRIBUTION = "laplace"  # options: ['laplace', 'gaussian']


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = (
    None  # None if val data from all scenes are bundled into a single npz file
)
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = (
    None  # None if test data from all scenes are bundled into a single npz file
)
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = (
    0.4  # discard data with overlap_score < min_overlap_score
)
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8


##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 4 * 8
_CN.TRAINER.CANONICAL_LR = 2e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.0  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = "linear"  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.0
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
# [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER = "MultiStepLR"
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"  # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
# ELR: ExponentialLR, this value for 'step' interval
_CN.TRAINER.ELR_GAMMA = 0.999992

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = False
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32  # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = "evaluation"  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = "dynamic"

# geometric metrics and pose solver
# recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.EPI_ERR_THR = 1e-4
_CN.TRAINER.POSE_GEO_MODEL = "E"  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = "RANSAC"  # [RANSAC, LO-RANSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = (
    "scene_balance"  # options: ['scene_balance', 'random', 'normal']
)
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
# whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True
# after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
