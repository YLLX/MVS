import os
import sys
import json
import shutil
import argparse
import datetime
import multiprocessing

import common
import parallel
import initialization
from config import cfg
from function import *
import pair_selection

import warnings
warnings.filterwarnings('ignore')


def main(user_cfg, start_from=0):
    """
    Launch the s2p pipeline with the parameters given in a json file.

    Args:
        user_cfg: user config dictionary
        start_from: the step to start from (default: 0)
    """
    common.print_elapsed_time.t0 = datetime.datetime.now()
    initialization.build_cfg(user_cfg)
    initialization.make_dirs()

    # multiprocessing setup
    nb_workers = multiprocessing.cpu_count()  # nb of available cores
    if cfg['max_processes'] is not None:
        nb_workers = cfg['max_processes']

    # pair selection
    # list_pairs_idx = pair_selection.select_pairs(cfg["images"], cfg['roi']['x'], cfg['roi']['y'], cfg['roi']['w'], cfg['roi']['h'], True)
    x, y, w, h = cfg['roi']['x'], cfg['roi']['y'], cfg['roi']['w'], cfg['roi']['h']
    ref_list, cluster_list = pair_selection.group_by_cluster(cfg['images'], x, y, w, h)
    list_cluster_pairs = pair_selection.get_tri_pairs(cfg['images'], ref_list, cluster_list, x, y, w, h)
    n_cluster = len(list_cluster_pairs)
    for i in range(n_cluster):
        with open(os.path.join(cfg['out_dir'], f"cluster_{i}_pairs.txt"), 'w') as f:
            f.writelines([f"{r} {s} {t}" for r,s,t in list_cluster_pairs[i]])

    cfg['refs'] = ref_list
    cfg['cluster_pairs'] = list_cluster_pairs
    cluster_imgs = []
    for pairs in list_cluster_pairs:
        cluster_imgs.append([pairs[0][0]] + list(set(sum([[j,k] for i,j,k in pairs], []))))
    cfg['cluster_imgs'] = cluster_imgs


    # adjust tile size
    tw, th = initialization.adjust_tile_size()

    for i in range(n_cluster):

        cfg['current_out_dir'] = os.path.join(cfg['out_dir'], f"cluster_{i}")
        cfg['current_cluster'] = i

        tiles_txt = os.path.join(cfg['current_out_dir'], 'tiles.txt')
        tiles = initialization.tiles_full_info(tw, th, tiles_txt, create_masks=True)
        # tiles: key(coordinates, dir, json, neighborhood_dirs)

        if not tiles:
            print('ERROR: the ROI is not seen in two images or is totally masked.')
            sys.exit(1)

        if start_from > 0:
            assert os.path.exists(tiles_txt), "start_from set to {} but tiles.txt is not found in '{}'. Make sure this is" \
                                            " the output directory of a previous run.".format(start_from, cfg['current_out_dir'])
        else:
            # initialisation: write the list of tilewise json files to outdir/tiles.txt
            with open(tiles_txt, 'w') as f:
                for t in tiles:
                    print(t['json'], file=f)

        # n = len(cfg['images'])
        # tiles_pairs = [(t, i) for i in range(1, n) for t in tiles]
        tiles_pairs = [(t, r, j, k) for r, j, k in cfg['cluster_pairs'][i] for t in tiles]
        timeout = cfg['timeout']

        # local-pointing step:
        if start_from <= 1:
            print('1) correcting pointing locally...')
            correction_pairs = [(t, cfg['cluster_imgs'][i][0], j) for j in cfg['cluster_imgs'][i][1:] for t in tiles]
            parallel.launch_calls(pointing_correction, correction_pairs, nb_workers, timeout=timeout)

        # global-pointing step:
        if start_from <= 2:
            print('2) correcting pointing globally...')
            global_pointing_correction(tiles)
            common.print_elapsed_time()

        # rectification step:
        if start_from <= 3:

            disp_file = os.path.join(cfg["out_dir"], "all_disp_min_max.txt")
            if(os.path.exists(disp_file)):
                os.remove(disp_file)
            alt_file = os.path.join(cfg["out_dir"], "all_altitude_range.txt")
            if(os.path.exists(alt_file)):
                os.remove(alt_file)

            print('3) rectifying tiles...')
            parallel.launch_calls(rectification_pair, tiles_pairs, nb_workers, timeout=timeout)

            with open(disp_file, 'r') as f:
                lines = [line.split() for line in f.readlines()]
            with open(disp_file, 'w') as f:
                f.writelines([" ".join(line) + "\n" for line in sorted(lines)])

            with open(alt_file, "r+") as f:
                lines = [line.split() for line in f.readlines()]
            with open(alt_file, 'w') as f:
                f.writelines([" ".join(line) + "\n" for line in sorted(lines)])


        # matching step:
        if start_from <= 4:
            print('4) running stereo matching...')
            if cfg['max_processes_stereo_matching'] is not None:
                nb_workers_stereo = cfg['max_processes_stereo_matching']
            else:
                nb_workers_stereo = nb_workers
            parallel.launch_calls(stereo_matching, tiles_pairs, nb_workers_stereo, timeout=timeout)

        if start_from <= 5:
            # triangulation step:
            print('5) triangulating tiles...')
            parallel.launch_calls(disparity_to_ply, tiles_pairs, nb_workers, timeout=timeout)


            # # disparity-to-height step:
            # print('5a) computing height maps...')
            # parallel.launch_calls(disparity_to_height, tiles_pairs, nb_workers, timeout=timeout)

            # print('5b) computing local pairwise height offsets...')
            # parallel.launch_calls(mean_heights, tiles, nb_workers, timeout=timeout)

            # # global-mean-heights step:
            # print('5c) computing global pairwise height offsets...')
            # global_mean_heights(tiles)

            # # heights-to-ply step:
            # print('5d) merging height maps and computing point clouds...')
            # parallel.launch_calls(heights_to_ply, tiles, nb_workers, timeout=timeout)



        # local-dsm-rasterization step:
        if start_from <= 6:
            print('computing DSM by tile...')
            parallel.launch_calls(plys_to_dsm, tiles_pairs, nb_workers, timeout=timeout)

        # global-dsm-rasterization step:
        if start_from <= 7:
            print('7) computing global DSM...')
            global_dsm(tiles_pairs)
        common.print_elapsed_time()

        # cleanup
        common.garbage_cleanup()
        common.print_elapsed_time(since_first_call=True)

def make_path_relative_to_file(path, f):
    return os.path.join(os.path.abspath(os.path.dirname(f)), path)

def read_config_file(config_file):
    """
    Read a json configuration file and interpret relative paths.

    If any input or output path is a relative path, it is interpreted as
    relative to the config_file location (and not relative to the current
    working directory). Absolute paths are left unchanged.

    如果config文件中的输入或者输出路径为相对路径, 相对于config文件的路径。
    如果为绝对路径，就不更改。
    """
    with open(config_file, 'r') as f:
        user_cfg = json.load(f)

    # output paths
    if not os.path.isabs(user_cfg['out_dir']):
        user_cfg['out_dir'] = make_path_relative_to_file(user_cfg['out_dir'], config_file)

    # 外部DEM path
    if 'exogenous_dem' in user_cfg and user_cfg['exogenous_dem'] is not None and not os.path.isabs(user_cfg['exogenous_dem']):
        user_cfg['exogenous_dem'] = make_path_relative_to_file(user_cfg['exogenous_dem'], config_file)

    # input paths (images and metadata)
    if not os.path.isabs(user_cfg['img_dir']):
        user_cfg['img_dir'] = make_path_relative_to_file(user_cfg['img_dir'], config_file)
    if not os.path.isabs(user_cfg['metadata_dir']):
        user_cfg['metadata_dir'] = make_path_relative_to_file(user_cfg['metadata_dir'], config_file)

    # 获取路径下图像
    # 如果存在 images 列表，‘img_dir’就失效
    if 'images' in user_cfg:
        for img in user_cfg['images']:
            for d in ['img', 'rpc', 'clr', 'cld', 'roi', 'wat']:
                if d in img and isinstance(img[d], str) and not os.path.isabs(img[d]):
                    img[d] = make_path_relative_to_file(img[d], config_file)
    else:
        user_cfg["images"] = []
        list_img = [os.path.join(user_cfg['img_dir'], f) for f in os.listdir(user_cfg['img_dir']) if f.endswith(('.tif', '.tiff'))]
        for i in range(len(list_img)):
            user_cfg["images"].append({"img": list_img[i]})

        # 如果存在单独的 RPC 文件（存放在 img_dir 路径下），将其添加到字典中
        list_rpc = [os.path.join(user_cfg['img_dir'], f) for f in os.listdir(user_cfg['img_dir']) if f.endswith(('.rpc', '.txt'))]
        if list_rpc:
            assert(len(list_img) == len(list_rpc))
            for i in range(len(user_cfg["images"])):
                user_cfg["images"][i]["rpc"] = list_rpc[i]

    return user_cfg


def cli():
    """
    Command line parsing for s2p command line interface.
    """
    parser = argparse.ArgumentParser(description=('S2P: Satellite Stereo Pipeline'))

    parser.add_argument('--config', default="./data/input/JAX_167/config.json",
                        help=('path to a json file containing the paths to input and '
                              'output files and the algorithm par ameters'))
    parser.add_argument('--start_from', dest='start_from', type=int, default=4,
                        help="Restart the process from a given step in "
                             "case of an interruption or to try different parameters.")
    args = parser.parse_args()

    user_cfg = read_config_file(args.config)

    main(user_cfg, start_from=args.start_from)

    # Backup input file for sanity check
    if not args.config.startswith(os.path.abspath(cfg['out_dir'] + os.sep)):
        shutil.copy2(args.config, os.path.join(cfg['out_dir'], 'config.json.orig'))



if __name__ == "__main__":
    cli()
