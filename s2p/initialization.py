from config import cfg
import rpc_utils
import geographiclib
import masking
import parallel
import common

import sys
import rpcm
import rasterio
import numpy as np
import os
import copy
import json
import random



# This function is here as a workaround to python bug #24313 When
# using python3, json does not know how to serialize numpy.int64 on
# some platform numpy also decides to go for int64 when numpy.arange
# is called. This results in our json not being serializable anymore
# Calling json.dump(..,default=workaround_json_int64) fixes this
# https://bugs.python.org/issue24313
def workaround_json_int64(o):
    if isinstance(o, np.integer) : return int(o)
    raise TypeError


def dict_has_keys(d, l):
    """
    Return True if the dict d contains all the keys of the input list l.
    """
    return all(k in d for k in l)

def check_parameters(d):
    """
    Check that the provided dictionary defines all mandatory s2p arguments.

    Args:
        d: python dictionary
    """
    # verify that input files paths are defined
    if 'images' not in d or len(d['images']) < 2:
        print('ERROR: missing paths to input images')
        sys.exit(1)
    for img in d['images']:
        if not dict_has_keys(img, ['img']):
            print('ERROR: missing img paths for image', img)
            sys.exit(1)

    # read RPCs：RPCs可以是单独的文件，也可以是嵌入在影像中。
    for img in d['images']:
        if 'rpc' in img:
            if isinstance(img['rpc'], str):  # path to an RPC file
                img['rpcm'] = rpcm.rpc_from_rpc_file(img['rpc'])
            elif isinstance(img['rpc'], dict):  # RPC dict in 'rpcm' format
                img['rpcm'] = rpcm.RPCModel(img['rpc'], dict_format='rpcm')
            else:
                raise NotImplementedError(
                    'rpc of type {} not supported'.format(type(img['rpc']))
                )
        else:
            img['rpcm'] = rpcm.rpc_from_geotiff(img['img'])

    # verify that an input ROI is defined
    # 如果定义了‘full_img’，‘roi’参数会被忽略掉
    if d.get("full_img"):
        with rasterio.open(d['images'][0]['img'], "r") as f:
            width = f.width
            height = f.height
        d['roi'] = {'x': 0, 'y': 0, 'w': width, 'h': height}
    elif 'roi' in d and dict_has_keys(d['roi'], ['x', 'y', 'w', 'h']):
        pass
    else:
        print('ERROR: missing or incomplete roi definition')
        sys.exit(1)

    # d['roi'] : all the values must be integers
    d['roi']['x'] = int(np.floor(d['roi']['x']))
    d['roi']['y'] = int(np.floor(d['roi']['y']))
    d['roi']['w'] = int(np.ceil(d['roi']['w']))
    d['roi']['h'] = int(np.ceil(d['roi']['h']))

    # warn about unknown parameters. The known parameters are those defined in
    # the global config.cfg dictionary, plus the mandatory 'images' and 'roi'
    for k in d.keys():
        if k not in ['images', 'metadata_dir', 'img_dir', 'out_dir', 'roi', "full_img"] and k not in cfg:
            print('WARNING: ignoring unknown parameter {}.'.format(k))


def build_cfg(user_cfg):
    """
    Populate a dictionary containing the s2p parameters from a user config file.

    This dictionary is contained in the global variable 'cfg' of the config
    module.

    Args:
        user_cfg: user config dictionary
    """
    # check that all the mandatory arguments are defined
    # 保证所有必须参数已经被定义, 主要是 rpc 和 roi
    check_parameters(user_cfg)

    # fill the config module: updates the content of the config.cfg dictionary
    # with the content of the user_cfg dictionary
    cfg.update(user_cfg)

    # set keys 'clr', 'cld', 'wat' and 'roi' of the reference image to None if they
    # are not already defined. The default values of these optional arguments
    # can not be defined directly in the config.py module. They would be
    # overwritten by the previous update, because they are in a nested dict.
    # 如果key没在字典中，setdefault才会设置默认value
    # 针对参考影像，但是 MVS 没有参考影像
    # cfg['images'][0].setdefault('clr', None)
    # cfg['images'][0].setdefault('cld', None)
    # cfg['images'][0].setdefault('roi', None)
    # cfg['images'][0].setdefault('wat', None)

    # get out_crs 坐标参考系统
    if 'out_crs' not in cfg or cfg['out_crs'] is None:
        x, y, w, h = [cfg['roi'][k] for k in ['x', 'y', 'w', 'h']]
        utm_zone = rpc_utils.utm_zone(cfg['images'][0]['rpcm'], x, y, w, h)
        epsg_code = geographiclib.epsg_code_from_utm_zone(utm_zone)
        cfg['out_crs'] = "epsg:{}".format(epsg_code)
        if cfg['out_geoid']:
            # Use the EGM96 geoid model for the output CRS if out_geoid is True
            cfg['out_crs'] += "+5773"

    # get image ground sampling distance
    cfg['gsd'] = rpc_utils.gsd_from_rpc(cfg['images'][0]['rpcm'])


def make_dirs():
    """
    Create directories needed to run s2p.
    """
    os.makedirs(cfg['out_dir'], exist_ok=True)
    os.makedirs(os.path.expandvars(os.path.join(cfg['out_dir'], cfg['temporary_dir'])), exist_ok=True)

    # store a json dump of the config.cfg dictionary
    with open(os.path.join(cfg['out_dir'], 'config.json'), 'w') as f:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['out_dir'] = '.'
        for img in cfg_copy['images']:
            img.pop('rpcm', None)
        json.dump(cfg_copy, f, indent=2, default=workaround_json_int64)


def adjust_tile_size():
    """
    Adjust the size of the tiles.
    """
    # 动态调整 tile_size 使得能够占满整个影像
    tile_w = min(cfg['roi']['w'], cfg['tile_size'])  # tile width
    ntx = int(np.round(float(cfg['roi']['w']) / tile_w))
    # ceil so that, if needed, the last tile is slightly smaller
    tile_w = int(np.ceil(float(cfg['roi']['w']) / ntx))

    tile_h = min(cfg['roi']['h'], cfg['tile_size'])  # tile height
    nty = int(np.round(float(cfg['roi']['h']) / tile_h))
    tile_h = int(np.ceil(float(cfg['roi']['h']) / nty))

    print('tile size: {} {}'.format(tile_w, tile_h))
    return tile_w, tile_h


def compute_tiles_coordinates(rx, ry, rw, rh, tw, th):
    """
    """
    out = []
    neighborhood_dict = dict()

    for y in np.arange(ry, ry + rh, th):
        for x in np.arange(rx, rx + rw, tw):
            out.append((x, y, tw, th))

            # get coordinates of tiles from neighborhood
            out2 = []
            for y2 in [y - th, y, y + th]:
                for x2 in [x - tw, x, x + tw]:
                    if rx + rw > x2 >= rx and ry + rh > y2 >= ry:
                        out2.append((x2, y2, tw, th))

            neighborhood_dict[str((x, y, tw, th))] = out2

    return out, neighborhood_dict

def get_tile_dir(x, y, w, h):
    """
    Get the name of a tile directory
    """
    return os.path.join('tiles','row_{:07d}_height_{}'.format(y, h),
                        'col_{:07d}_width_{}'.format(x, w))


def create_tile(coords, neighborhood_coords_dict):
    """
    Return a dictionary with the data of a tile.

    Args:
        coords (tuple): 4-tuple of ints giving the x, y, w, h coordinates of a
            tile, where x, y are the top-left corner coordinates and w, h the
            width and height
        neighborhood_coords_dict (dict): dictionary with the list of
            neighboring tiles of each tile. The keys of this dict are string
            identifying the tiles, and the values are lists of tuples of
            coordinates of neighboring tiles coordinates

    Returns:
        tile (dict): dictionary with the metadata of a tile
    """
    tile = {}
    tile['coordinates'] = coords
    tile['dir'] = os.path.join(cfg['current_out_dir'], get_tile_dir(*coords))
    tile['json'] = os.path.join(get_tile_dir(*coords), 'config.json')

    tile['neighborhood_dirs'] = list()
    key = str(coords)

    if 'neighborhood_dirs' in cfg:
        tile['neighborhood_dirs'] = cfg['neighborhood_dirs']
    elif key in neighborhood_coords_dict:
        for coords2 in neighborhood_coords_dict[key]:
            tile['neighborhood_dirs'].append(os.path.join('../../..', get_tile_dir(*coords2)))

    return tile


def rectangles_intersect(r, s):
    """
    Check intersection of two rectangles parallel to the coordinate axis.

    Args:
        r (tuple): 4 floats that define the coordinates of the top-left corner,
            the width and the height of a rectangle
        s (tuple): 4 floats that define the coordinates of the top-left corner,
            the width and the height of a rectangle

    Return:
        bool telling if the rectangles intersect
    """
    rx, ry, rw, rh = r
    sx, sy, sw, sh = s

    # check if one rectangle is entirely above the other
    if ry + rh < sy or sy + sh < ry:
        return False

    # check if one rectangle is entirely left of the other
    if rx + rw < sx or sx + sw < rx:
        return False

    return True

def is_tile_all_nodata(path:str, window:rasterio.windows.Window):
    """Check if pixels in a given window are all nodata.

    Parameters
    ----------
    path
        Path to the raster.
    window
        A rasterio.windows.Window object.

    Returns
    -------
        Return True if all pixels in the window are nodata.
        Return False if at least one pixel is non-nodata.
    """
    with rasterio.open(path, "r") as ds:
        arr = ds.read(window=window)

        # NOTE: Many satellite imagery providers use ds.nodata as the value of
        # nodata pixels. Pleiades and PNeo imagery use None as nodata in their
        # profile while putting 0 to nodata pixel in reality. Thus, we have to
        # check both ds.nodata and 0 here. I.e., if a window is full of nodata
        # or 0, then this window is discarded.
        if np.all(arr == 0) or np.all(arr == ds.nodata):
            return True
        else:
            return False


def is_this_tile_useful(x, y, w, h, images_sizes):
    """
    Check if a tile contains valid pixels.

    Valid pixels must be found in the reference image plus at least one other image.

    Args:
        x, y, w, h (ints): 4 ints that define the coordinates of the top-left corner,
            the width and the height of a rectangular tile
        images_sizes (list): list of tuples with the height and width of the images

    Return:
        useful (bool): bool telling if the tile has to be processed
        mask (np.array): tile validity mask. Set to None if the tile is discarded
    """
    ref_idx = cfg['cluster_imgs'][cfg['current_cluster']][0]
    if is_tile_all_nodata(cfg["images"][ref_idx]["img"], rasterio.windows.Window(x, y, w, h)):
        return False, None

    # check if the tile is partly contained in at least one other image
    rpc = cfg['images'][ref_idx]['rpcm']
    for idx, size in zip(cfg['cluster_imgs'][1:], images_sizes[1:]):
        coords = rpc_utils.corresponding_roi(rpc, cfg['images'][idx]['rpcm'], x, y, w, h)
        if rectangles_intersect(coords, (0, 0, size[1], size[0])):
            break  # the tile is partly contained
    else:  # we've reached the end of the loop hence the tile is not contained
        return False, None

    roi_msk = cfg['images'][ref_idx]['roi']
    cld_msk = cfg['images'][ref_idx]['cld']
    wat_msk = cfg['images'][ref_idx]['wat']
    mask = masking.image_tile_mask(x, y, w, h, roi_msk, cld_msk, wat_msk,
                                   images_sizes[0], cfg['border_margin'])
    if not mask.any():
        return False, None
    return True, mask

def tiles_full_info(tw, th, tiles_txt, create_masks):
    """
    List the tiles to process and prepare their output directories structures.

    Most of the time is spent discarding tiles that are masked by water
    (according to exogenous dem).

    Returns:
        a list of dictionaries. Each dictionary contains the image coordinates
        and the output directory path of a tile.
    """
    ref_idx = cfg['cluster_imgs'][cfg['current_cluster']][0]

    cfg['images'][ref_idx].setdefault('clr', None)
    cfg['images'][ref_idx].setdefault('cld', None)
    cfg['images'][ref_idx].setdefault('roi', None)
    cfg['images'][ref_idx].setdefault('wat', None)

    rpc = cfg['images'][ref_idx]['rpcm']
    roi_msk = cfg['images'][ref_idx]['roi']
    cld_msk = cfg['images'][ref_idx]['cld']
    wat_msk = cfg['images'][ref_idx]['wat']

    rx = cfg['roi']['x']
    ry = cfg['roi']['y']
    rw = cfg['roi']['w']
    rh = cfg['roi']['h']

    # list of dictionaries (one for each non-masked tile)
    tiles = []

    # list tiles coordinates
    tiles_coords, neighborhood_coords_dict = compute_tiles_coordinates(rx, ry, rw, rh, tw, th)

    if create_masks or not os.path.exists(tiles_txt):
        print('\ndiscarding masked tiles...')
        images_sizes = []
        for i in cfg['cluster_imgs'][cfg['current_cluster']]:
            with rasterio.open(cfg['images'][i]['img'], 'r') as f:
                images_sizes.append(f.shape)

        # compute all masks in parallel as numpy arrays
        tiles_usefulnesses = parallel.launch_calls(is_this_tile_useful,
                                                   tiles_coords,
                                                   cfg['max_processes'],
                                                   images_sizes,
                                                   tilewise=False,
                                                   timeout=cfg['timeout'])

        # discard useless tiles from neighborhood_coords_dict
        discarded_tiles = set(x for x, (b, _) in zip(tiles_coords, tiles_usefulnesses) if not b)
        for k, v in neighborhood_coords_dict.items():
            neighborhood_coords_dict[k] = list(set(v) - discarded_tiles)

        for coords, usefulness in zip(tiles_coords, tiles_usefulnesses):

            useful, mask = usefulness
            if not useful:
                continue

            tile = create_tile(coords, neighborhood_coords_dict)
            tiles.append(tile)

            # make tiles directories and store json configuration dumps
            os.makedirs(tile['dir'], exist_ok=True)
            for ref_idx, sec_idx, trd_idx in cfg["cluster_pairs"][cfg['current_cluster']]:
                os.makedirs(os.path.join(tile['dir'], f'pair_{ref_idx}_{sec_idx}_{trd_idx}'), exist_ok=True)

            # save a json dump of the tile configuration
            tile_cfg = copy.deepcopy(cfg)
            x, y, w, h = tile['coordinates']
            for img in tile_cfg['images']:
                img.pop('rpcm', None)
            tile_cfg['roi'] = {'x': x, 'y': y, 'w': w, 'h': h}
            tile_cfg['full_img'] = False
            tile_cfg['max_processes'] = 1
            tile_cfg['neighborhood_dirs'] = tile['neighborhood_dirs']
            tile_cfg['out_dir'] = '../../..'

            with open(os.path.join(cfg['current_out_dir'], tile['json']), 'w') as f:
                json.dump(tile_cfg, f, indent=2, default=workaround_json_int64)

            # save the mask
            common.rasterio_write(os.path.join(tile['dir'], 'mask.tif'), mask.astype(np.uint8))
    else:
        if len(tiles_coords) == 1:
            tiles.append(create_tile(tiles_coords[0], neighborhood_coords_dict))
        else:
            with open(tiles_txt, 'r') as f_tiles:
                for config_json in f_tiles:
                    tile = {}
                    with open(os.path.join(cfg['current_out_dir'], config_json.rstrip(os.linesep)), 'r') as f_config:
                        tile_cfg = json.load(f_config)
                        roi = tile_cfg['roi']
                        coords = roi['x'], roi['y'], roi['w'], roi['h']
                        tiles.append(create_tile(coords, neighborhood_coords_dict))

    return tiles
