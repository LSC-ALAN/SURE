import argparse
import os
import time
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SURE on a single image pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image0",
        default="assets/scannet_sample_images/scene0707_00_15.jpg",
        help="path to the first image",
    )
    parser.add_argument(
        "--image1",
        default="assets/scannet_sample_images/scene0707_00_45.jpg",
        help="path to the second image",
    )
    parser.add_argument("--ckpt_path", default="SURE.ckpt", help="checkpoint path")
    parser.add_argument(
        "--main_cfg_path",
        default="configs/sure/indoor/sure_base.py",
        help="model config path",
    )
    parser.add_argument(
        "--data_cfg_path",
        default="configs/data/scannet_test_1500.py",
        help="data config path",
    )
    parser.add_argument("--width", type=int, default=640, help="input width")
    parser.add_argument("--height", type=int, default=480, help="input height")
    parser.add_argument("--topk", type=int, default=None, help="override coarse top-k")
    parser.add_argument(
        "--mconf_thr", type=float, default=0.2, help="coarse matching confidence threshold"
    )
    parser.add_argument("--border_rm", type=int, default=2, help="border removal")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs")
    parser.add_argument("--repeat", type=int, default=1, help="timed runs")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="inference device",
    )
    parser.add_argument(
        "--output",
        default="sure_single_pair.png",
        help="path for the visualization image",
    )
    return parser.parse_args()


def read_image(cv2, path, width, height):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (width, height))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    args = parse_args()

    import cv2
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import torch

    from src.config.default import get_cfg_defaults
    from src.sure.sure import SURE
    from src.utils.misc import lower_config
    from src.utils.plotting import make_matching_figure

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    device = torch.device(args.device)

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    config.SURE.TEST_RES_W = args.width
    config.SURE.TEST_RES_H = args.height
    config.SURE.COARSE.MCONF_THR = args.mconf_thr
    config.SURE.COARSE.BORDER_RM = args.border_rm
    config.SURE.COARSE.TOPK = (
        args.topk if args.topk is not None else int(args.width / 8 * args.height / 8 * 0.35)
    )

    matcher = SURE(config=lower_config(config)["sure"]).to(device)
    state_dict = torch.load(args.ckpt_path, map_location=device)["state_dict"]
    matcher.load_state_dict(state_dict)
    matcher.eval()

    img0_raw = read_image(cv2, args.image0, args.width, args.height)
    img1_raw = read_image(cv2, args.image1, args.width, args.height)

    img0 = torch.from_numpy(img0_raw).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    img1 = torch.from_numpy(img1_raw).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    batch = {"image0": img0, "image1": img1}

    with torch.no_grad():
        for _ in range(args.warmup):
            matcher(batch)

        synchronize(device)
        for _ in range(args.repeat):
            matcher(batch)
        synchronize(device)

    print(f"matches: {batch['mkpts0_f'].shape[0]}")

    mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
    mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()
    epistemic_mean = batch["epistemic_mean"].detach().cpu().numpy()
    color = cm.turbo(epistemic_mean)

    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=[])
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
