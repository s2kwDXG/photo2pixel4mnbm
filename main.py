import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2  # OpenCVをインポート
import argparse
import os

from models.module_photo2pixel import Photo2PixelModel
from utils import img_common_util


def convert(image_path, output_dir, kernel_size=4, pixel_size=10, edge_thresh=2048):
    # 出力ファイル名を生成
    base_name = os.path.basename(image_path)
    file_name, file_extension = os.path.splitext(base_name)
    output_file_name = f"{file_name}_k{kernel_size}_p{pixel_size}_e{edge_thresh}{file_extension}"
    output_path = os.path.join(output_dir, output_file_name)

    img_input = Image.open(image_path).convert("RGBA")
    img_np = np.array(img_input).astype(np.float32)
    img_rgb = img_np[:, :, :3]
    alpha_channel = img_np[:, :, 3]

    # サイズが異なる場合はOpenCVを使用してリサイズ
    if img_rgb.shape[:2] != alpha_channel.shape[:2]:
        alpha_channel = cv2.resize(alpha_channel, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    img_pt = np.transpose(img_rgb, axes=[2, 0, 1])[np.newaxis, :, :, :]  # RGBのみをテンソルに変換
    img_pt = torch.from_numpy(img_pt)
    alpha_channel = torch.from_numpy(alpha_channel).unsqueeze(0).unsqueeze(0)  # アルファチャンネルの次元を調整

    print(f"Processing {image_path}")
    print("Original RGB shape:", img_pt.shape)
    print("Original Alpha shape:", alpha_channel.shape)

    model = Photo2PixelModel()
    model.eval()
    with torch.no_grad():
        result_rgb_pt, result_alpha_pt = model(
            img_pt,
            alpha_channel,
            param_kernel_size=kernel_size,
            param_pixel_size=pixel_size,
            param_edge_thresh=edge_thresh
        )

    result_rgb_np = result_rgb_pt[0, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    result_alpha_np = result_alpha_pt[0, 0, :, :].cpu().numpy().astype(np.uint8)

    print("Processed RGB shape:", result_rgb_np.shape)
    print("Processed Alpha shape:", result_alpha_np.shape)

    # 処理後のサイズが異なる場合は再度OpenCVを使用してリサイズ
    if result_rgb_np.shape[:2] != result_alpha_np.shape[:2]:
        result_alpha_np = cv2.resize(result_alpha_np, (result_rgb_np.shape[1], result_rgb_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # RGBとアルファチャンネルを結合して画像を保存
    result_rgba_np = np.concatenate((result_rgb_np, result_alpha_np[:, :, np.newaxis]), axis=2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Image.fromarray(result_rgba_np).save(output_path)
    print(f"Saved to {output_path}")


def find_png_images(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                png_files.append(os.path.join(root, file))
    return png_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='algorithm converting photo to pixel art')
    parser.add_argument('--input', type=str, required=True, help='input image or directory path')
    parser.add_argument('--output', type=str, required=True, help='output directory path')
    parser.add_argument('--kernel_size', type=int, default=4, help='larger kernel size means smooth color transition')
    parser.add_argument('--pixel_size', type=int, default=10, help='individual pixel size')
    parser.add_argument('--edge_thresh', type=int, default=2048, help='lower edge threshold means more black line in edge region')
    args = parser.parse_args()

    # 入力がディレクトリの場合は再帰的にPNG画像を取得
    if os.path.isdir(args.input):
        png_images = find_png_images(args.input)
        if not png_images:
            print(f"No PNG images found in directory {args.input}")
            exit(1)
    else:
        png_images = [args.input]

    # 出力ディレクトリを作成
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 各画像を処理
    for image_path in png_images:
        convert(image_path, args.output, kernel_size=args.kernel_size, pixel_size=args.pixel_size, edge_thresh=args.edge_thresh)
