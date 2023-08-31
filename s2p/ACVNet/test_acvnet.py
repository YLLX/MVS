from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

from ACVNet.acv import ACVNet
from ACVNet.tiling import tile_image, untile_image
from ACVNet.data_io import load_image, get_transform

from imsat_tools.libTP.stereo import mismatchFiltering


import skimage
from tqdm import tqdm




def test(left, right, model, using_cuda=False):

    left = Variable(left, requires_grad=False)
    right = Variable(right, requires_grad=False)

    model.eval()
    if using_cuda:
        left = left.cuda()
        right = right.cuda()
    with torch.no_grad():
        pred = model(left, right)
        pred = pred[-1].cpu().detach().numpy()
        return pred




def main(args=[]):

    parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
    parser.add_argument('--maxdisp', type=int, default=96, help='maximum disparity')
    parser.add_argument('--mindisp', type=int, default=0, help='minimum disparity')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--loadckpt', default=os.path.join(current_dir, './pretrained_model/us3d_step03.ckpt'),
                                    help='load the weights from a specific checkpoint')

    parser.add_argument('--rectified_left', type=str)
    parser.add_argument('--rectified_right', type=str)
    parser.add_argument('--rectified_disp', type=str)
    parser.add_argument('--crop_height', default=512, type=int)
    parser.add_argument('--crop_width', default=512, type=int)

    parser.add_argument('--do_mismatch_filtering', default=False)
    parser.add_argument('--stereo_speckle_filter', default=20)

    parser.add_argument('--using_cuda', default=True)

    # parse arguments
    args = parser.parse_args(args)

    # model
    model = ACVNet(args.mindisp, args.maxdisp, False, False)
    model = torch.nn.DataParallel(model)

    #load parameters
    print("loading model {}".format(args.loadckpt))
    if ~torch.cuda.is_available():
        args.using_cuda = False
    if args.using_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    state_dict = torch.load(args.loadckpt, map_location=device)
    model.load_state_dict(state_dict['model'])

    # load images
    left = load_image(args.rectified_left)
    right = load_image(args.rectified_right)

    # transform
    processed = get_transform()
    left = processed(left)
    right = processed(right)
    assert left.shape == right.shape

    left_tiles, tile_origins = tile_image(left, args.crop_width, args.crop_height)
    right_tiles, tile_origins = tile_image(right, args.crop_width, args.crop_height)

    print('ACVNet: computing LR disparity')
    pred_tiles = test(left_tiles, right_tiles, model, args.using_cuda)
    disp_lr = untile_image(pred_tiles, tile_origins)

    if args.do_mismatch_filtering:
        print('ACVNet: computing RL disparity')

        left_flipped = torch.fliplr(left)
        right_flipped = torch.fliplr(right)

        left_tiles, tile_origins = tile_image(right_flipped, args.crop_width, args.crop_height)
        right_tiles, tile_origins = tile_image(left_flipped, args.crop_width, args.crop_height)

        pred_tiles = test(left_tiles, right_tiles, model, args.using_cuda)
        disp_rl = untile_image(pred_tiles, tile_origins)

        disp_lrs = mismatchFiltering(disp_lr, -np.fliplr(disp_rl), args.stereo_speckle_filter)
    else:
        disp_lrs = disp_lr


    # save filenames
    if args.do_mismatch_filtering:
        disp_filename_no_extension, extension = os.path.splitext(args.rectified_disp)
        unfiltered_lr_filename = disp_filename_no_extension + '_lr' + extension
        unfiltered_rl_filename = disp_filename_no_extension + '_rl' + extension
        skimage.io.imsave(unfiltered_lr_filename, -disp_lr)
        skimage.io.imsave(unfiltered_rl_filename, -disp_rl)
    skimage.io.imsave(args.rectified_disp, -disp_lrs)
