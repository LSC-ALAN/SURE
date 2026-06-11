import lightning.pytorch as pl
import argparse
import pprint
import os
import sys
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_sure import PL_SURE

import torch


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # data_cfg_path = "configs/data/megadepth_test_1500.py"
    # main_cfg_path = "configs/sure/outdoor/sure_base.py"
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/sure/outdoor/sure_base.py")
    parser.add_argument(
        '--ckpt_path', type=str, default="SURE.ckpt", help='path to the checkpoint')
    # parser.add_argument(
    #     '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    # parser.add_argument(
    #     '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=0)
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="ddp")

    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="if set, the matching results will be dump to dump_dir",
    )
    parser.add_argument("--benchmark", default=True)
    parser.add_argument(
        "--profiler_name",
        type=str,
        default="inference",
        help="options: [inference, pytorch], or leave it unset",
    )
    parser.add_argument(
        "--deter",
        action="store_true",
        default=False,
        help="use deterministic mode for testing",
    )
    parser.add_argument(
        "--pixel_thr", type=float, default=None, help="modify the RANSAC threshold."
    )
    parser.add_argument(
        "--ransac", type=str, default=None, help="modify the RANSAC method"
    )
    parser.add_argument(
        "--W", type=int, default=None, help="image width"
    )
    parser.add_argument(
        "--H", type=int, default=None, help="image height"
    )


    return parser.parse_args()

if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.deter:
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    if args.pixel_thr is not None:
        config.TRAINER.RANSAC_PIXEL_THR = args.pixel_thr

    if args.ransac is not None:
        config.TRAINER.POSE_ESTIMATION_METHOD = args.ransac

    if args.W is not None and args.H is not None:
        config.SURE.TEST_RES_W = args.W
        config.SURE.TEST_RES_H = args.H
        config.SURE.COARSE.TOPK = int(args.W / 8 * args.H / 8 * 0.35)

    loguru_logger.info(f"Args and config initialized!")

    assert args.ckpt_path is not None
    if not os.path.exists(args.ckpt_path):
        print("check input ckpt_path.")
        sys.exit(1)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_SURE(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        # dump_dir=args.dump_dir,
    )
    loguru_logger.info(f"SURE-lightning initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp",
        num_nodes=args.num_nodes,
        benchmark=True,
        logger=False,
        use_distributed_sampler=False,
        profiler=profiler,
    )

    loguru_logger.info(f"Start testing!")

    trainer.test(model, datamodule=data_module, verbose=False)
