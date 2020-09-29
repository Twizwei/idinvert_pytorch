"""
Generate a zoom-in or a translation animation. 
    TODO: now we can only start from the first frame!
"""
import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from zoom_py import transform

if __name__ == '__main__':
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='./fake_frames',
                        help='Directory to save the results. If not specified, ')
    parser.add_argument('--input_path', type=str, default=None,
                        help='input path to the image that you want to zoom in')
    parser.add_argument('--zoom_scale', type=float, default=None,
                        help='zoom-in scale')
    parser.add_argument('--translation', type=float, default=None,
                        help='translation')
    args = parser.parse_args()

    # mkdir
    output_dir = args.output_dir
    output_frames_dir = os.path.join(args.output_dir, 'frames')
    output_flows_dir = os.path.join(args.output_dir, 'flows')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    os.makedirs(output_flows_dir, exist_ok=True)

    # load image
    if args.input_path is None:
        from skimage import data
        input_frame = data.coffee()  # ndarray
    else:
        input_frame = plt.imread(args.input_path)
    input_frame = np.transpose(input_frame, (2, 0, 1))  # to PyTorch format (C, H, W)
    input_frame = torch.from_numpy(input_frame)  # to tensor

    if args.zoom_scale is not None:
        zoom_in_scales = list(np.linspace(1.0, args.zoom_scale, num=2))
    else:
        zoom_in_scales = [1.0]

    if args.translation is not None:
        translations = np.arange(0, args.translation)
    else:
        translations = [0]

    # operate transform: zoom-in
    if args.zoom_scale is not None:
        print(input_frame.shape)
        plt.imsave(os.path.join(output_frames_dir, 'zoom_000.png'), input_frame.numpy().transpose(1, 2, 0))
        for itr_zoom, zoom_scale in enumerate(tqdm(zoom_in_scales)):
            transformed_im, transformed_flow = transform(input_frame, scale=zoom_scale, translation=(0, 8))
            transformed_im = np.transpose(transformed_im.squeeze().numpy(), (1, 2, 0)).astype(np.uint8) # to image
            plt.imsave(os.path.join(output_frames_dir, 'zoom_{0:03d}.png').format(itr_zoom + 1), transformed_im)  # save image
            np.save(os.path.join(output_flows_dir, 'zoom_{0:03d}.npy').format(itr_zoom + 1),
                    np.transpose(transformed_flow.squeeze().numpy(), (1, 2, 0)))  # save flow

    # operate transform: translation
    if args.translation is not None:
        plt.imsave(os.path.join(output_frames_dir, 'trans_000.png'), input_frame.numpy().transpose(1, 2, 0))
        for itr_trans, trans in enumerate(tqdm(translations)):
            transformed_im, transformed_flow = transform(input_frame, scale=1.0, translation=(0, trans))
            transformed_im = np.transpose(transformed_im.squeeze().numpy(), (1, 2, 0)).astype(np.uint8)  # to image
            plt.imsave(os.path.join(output_frames_dir, 'trans_{0:03d}.png').format(itr_trans + 1), transformed_im)  # save image
            np.save(os.path.join(output_flows_dir, 'trans_{0:03d}.npy').format(itr_trans + 1),
                    np.transpose(transformed_flow.squeeze().numpy(), (1, 2, 0)))  # save flow


    print('done!')

