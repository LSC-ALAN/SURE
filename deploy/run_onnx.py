import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import torch
import cv2
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt


H, W = 480, 640


def read_gray(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (W, H))
    image = image[None, None,] / 255.
    return image.astype(np.float32)


def read_rgb(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    image = image[:, :, ::-1]
    return image


def main():
    img0_pth = "../assets/scannet_sample_images/scene0707_00_15.jpg"
    img1_pth = "../assets/scannet_sample_images/scene0707_00_45.jpg"
    img0 = read_gray(img0_pth)
    img1 = read_gray(img1_pth)
    data = np.concatenate([img0, img1], 1)
    print(data.shape)

    options = {}

    onnx_model_path = "sure_w640_h480_topk1680.onnx"
    print("loading onnx model", onnx_model_path)
    session = ort.InferenceSession(
        onnx_model_path,
        providers=[('TensorrtExecutionProvider', options)]
    )


    input_name = session.get_inputs()[0].name

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_ms = 0
    warmup_num = 10
    test_num = 100
    print("warmup...")
    for _ in range(warmup_num):
        outputs = session.run(None, {input_name: data})
    print(f"start run {test_num} times...")
    torch.cuda.synchronize()
    for i in range(test_num):
        start_event.record()
        outputs = session.run(None, {input_name: data})
        end_event.record()
        torch.cuda.synchronize()
        total_ms += start_event.elapsed_time(end_event)

    print("avg runtime: ", total_ms / test_num)
    
    
    # Post processing
    output = outputs[0]
    print("Output shape:", output.shape) # [K ,11]
    # default use bi_directional_refine=True

    print("drawing...")
    kpts0_c = output[:, :2]
    kpts1_c = output[:, 2:4]
    fine_offset01 = output[:, 4:6]
    fine_offset10 = output[:, 6:8]
    pred_score01 = output[:, 8]
    pred_score10 = output[:, 9]
    mconf = output[:, 10]
    
    # Filter by mconf and border
    mconf_thr = 0.2
    sigma_thr = 1e-6
    local_resolution = 8
    border_rm = local_resolution * 2

    mkpts0_f = kpts0_c
    mkpts1_f = kpts1_c + fine_offset01*local_resolution
    mkpts0_f_ = kpts0_c + fine_offset10*local_resolution
    mkpts1_f_ = kpts1_c
    
    mkpts0_f = np.concatenate([mkpts0_f, mkpts0_f_])
    mkpts1_f = np.concatenate([mkpts1_f, mkpts1_f_])
    pred_score = np.concatenate([pred_score01, pred_score10])
    mconf = np.concatenate([mconf, mconf])

    mask = (
        (mconf > mconf_thr)
        & (mkpts0_f[:, 0] >= border_rm)
        & (mkpts0_f[:, 0] <= W - border_rm)
        & (mkpts0_f[:, 1] >= border_rm)
        & (mkpts0_f[:, 1] <= H - border_rm)
        & (mkpts1_f[:, 0] >= border_rm)
        & (mkpts1_f[:, 0] <= W - border_rm)
        & (mkpts1_f[:, 1] >= border_rm)
        & (mkpts1_f[:, 1] <= H - border_rm)
    )


    # Filter by sigma
    # Retain the more confident matching pair with a smaller sigma (more significant) in the bi-directional matching pairs
    pred_score_mask = pred_score01 > pred_score10
    pred_score_mask = np.concatenate([pred_score_mask, ~pred_score_mask])
    pred_score_mask &= pred_score > sigma_thr
    mask &= pred_score_mask

    mkpts0 = mkpts0_f[mask]
    mkpts1 = mkpts1_f[mask]
    mconf = mconf[mask]
    
    # Draw
    color = cm.jet(mconf)
    text = [
        'SURE',
        'Matches: {}'.format(len(mkpts0)),
    ]
    img0_rgb = read_rgb(img0_pth)
    img1_rgb = read_rgb(img1_pth)
    fig = make_matching_figure(
        img0_rgb, img1_rgb, mkpts0, mkpts1, color, text=text)
    plt.show()

if __name__ == "__main__":
    main()
