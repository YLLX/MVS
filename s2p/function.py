import os
import rasterio
import rasterio.merge
import numpy as np
from plyfile import PlyData, PlyElement

import ply
import fusion
import common
import masking
import triangulation
import geographiclib
import rectification
import visualisation
import block_matching
from config import cfg
import pointing_accuracy
from plyflatten import plyflatten_from_plyfiles_list


def pointing_correction(tile, i, j):
    """
    Compute the translation that corrects the pointing error on a pair of tiles.

    Args:
        tile: dictionary containing the information needed to process the tile
        i: index of the processed pair
    """
    x, y, w, h = tile['coordinates']
    out_dir = os.path.join(tile['dir'], 'correction')
    img1 = cfg['images'][i]['img']
    rpc1 = cfg['images'][i]['rpcm']
    img2 = cfg['images'][j]['img']
    rpc2 = cfg['images'][j]['rpcm']

    # correct pointing error
    print(f'correcting pointing on tile {x} {y} pair {i}_{j}...')

    method = 'relative' if cfg['relative_sift_match_thresh'] is True else 'absolute'
    A, m = pointing_accuracy.compute_correction(
        img1, img2, rpc1, rpc2, x, y, w, h, method,
        cfg['sift_match_thresh'], cfg['max_pointing_error']
    )

    if A is not None:  # A is the correction matrix
        np.savetxt(os.path.join(out_dir, f'pointing_{j}.txt'), A, fmt='%6.3f')
    if m is not None:  # m is the list of sift matches
        np.savetxt(os.path.join(out_dir, f'sift_matches_{j}.txt'), m, fmt='%9.3f')
        np.savetxt(os.path.join(out_dir, f'center_keypts_sec_{j}.txt'),
                   np.mean(m[:, 2:], 0), fmt='%9.3f')
        if cfg['debug']:
            visualisation.plot_matches(img1, img2, rpc1, rpc2, m,
                                       os.path.join(out_dir, f'sift_matches_pointing_{j}.png'),
                                       x, y, w, h)


def global_pointing_correction(tiles):
    """
    Compute the global pointing corrections for each pair of images.

    Args:
        tiles: list of tile dictionaries
    """
    for i in cfg['cluster_imgs'][1:]:
        out = os.path.join(cfg['current_out_dir'], f'global_pointing_{i}.txt')
        l = [os.path.join(t['dir'], 'correction') for t in tiles]
        np.savetxt(out, pointing_accuracy.global_from_local(l, i), fmt='%12.6f')
        if cfg['clean_intermediate']:
            for d in l:
                common.remove(os.path.join(d, f'center_keypts_sec_{i}.txt'))

def rectification_pair(tile, i, j):
    """
    Rectify a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], f'pair_{i}_{j}')
    x, y, w, h = tile['coordinates']
    img1 = cfg['images'][i]['img']
    rpc1 = cfg['images'][i]['rpcm']
    img2 = cfg['images'][j]['img']
    rpc2 = cfg['images'][j]['rpcm']
    pointing = os.path.join(cfg['out_dir'], f'global_pointing_pair_{i}_{j}.txt')

    print('rectifying tile {} {} pair {}_{}...'.format(x, y, i, j))
    try:
        A = np.loadtxt(os.path.join(out_dir, 'pointing.txt'))
    except IOError:
        A = np.loadtxt(pointing)
    try:
        m = np.loadtxt(os.path.join(out_dir, 'sift_matches.txt'))
    except IOError:
        m = None

    cur_dir = os.path.join(tile['dir'], 'pair_{}_{}'.format(i, j))
    for n in tile['neighborhood_dirs']:
        nei_dir = os.path.join(tile['dir'], n, 'pair_{}_{}'.format(i, j))
        if os.path.exists(nei_dir) and not os.path.samefile(cur_dir, nei_dir):
            sift_from_neighborhood = os.path.join(nei_dir, 'sift_matches.txt')
            try:
                m_n = np.loadtxt(sift_from_neighborhood)
                # added sifts in the ellipse of semi axes : (3*w/4, 3*h/4)
                m_n = m_n[np.where(np.linalg.norm([(m_n[:, 0] - (x + w/2)) / w,
                                                   (m_n[:, 1] - (y + h/2)) / h],
                                                  axis=0) < 3/4)]
                if m is None:
                    m = m_n
                else:
                    m = np.concatenate((m, m_n))
            except IOError:
                print('%s does not exist' % sift_from_neighborhood)

    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    H1, H2, disp_min, disp_max = rectification.rectify_pair(img1, img2,
                                                            rpc1, rpc2,
                                                            x, y, w, h,
                                                            rect1, rect2, A, m,
                                                            method=cfg['rectification_method'],
                                                            hmargin=cfg['horizontal_margin'],
                                                            vmargin=cfg['vertical_margin'])
    np.savetxt(os.path.join(out_dir, 'H_ref.txt'), H1, fmt='%12.6f')
    np.savetxt(os.path.join(out_dir, 'H_sec.txt'), H2, fmt='%12.6f')
    np.savetxt(os.path.join(out_dir, 'disp_min_max.txt'), [disp_min, disp_max], fmt='%3.1f')

    with open(os.path.join(cfg["out_dir"], 'all_disp_min_max.txt'), '+a') as f:
        f.write(f"tile({x}_{y})_pair({i}_{j}):\t {str(int(disp_min))} {str(int(disp_max))}\n")

    if cfg['clean_intermediate']:
        common.remove(os.path.join(out_dir, 'pointing.txt'))
        common.remove(os.path.join(out_dir, 'sift_matches.txt'))



def stereo_matching(tile, i, j):
    """
    Compute the disparity of a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}_{}'.format(i, j))
    x, y = tile['coordinates'][:2]

    print('estimating disparity on tile {} {} pair {}_{}...'.format(x, y, i, j))
    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.tif')
    disp_min, disp_max = np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))

    block_matching.compute_disparity_map(rect1, rect2, disp, mask, cfg['matching_algorithm'],
                                        disp_min, disp_max, timeout=cfg['mgm_timeout'],
                                        max_disp_range=cfg['max_disp_range'])
    # add margin around masked pixels
    masking.erosion(mask, mask, cfg['msk_erosion'])

    if cfg['clean_intermediate']:
        if len(cfg['images']) > 2:
            common.remove(rect1)
        common.remove(rect2)
        common.remove(os.path.join(out_dir, 'disp_min_max.txt'))


def disparity_to_height(tile, i, j):
    """
    Compute a height map from the disparity map of a pair of image tiles.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair.
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}_{}'.format(i, j))
    x, y, w, h = tile['coordinates']

    print('triangulating tile {} {} pair {}_{}...'.format(x, y, i, j))
    rpc1 = cfg['images'][i]['rpcm']
    rpc2 = cfg['images'][j]['rpcm']
    H_ref = np.loadtxt(os.path.join(out_dir, 'H_ref.txt'))
    H_sec = np.loadtxt(os.path.join(out_dir, 'H_sec.txt'))
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.tif')
    mask_orig = os.path.join(tile['dir'], 'mask.tif')
    pointing = os.path.join(cfg['out_dir'], 'global_pointing_pair_{}_{}.txt'.format(i, j))

    with rasterio.open(disp, 'r') as f:
        disp_img = f.read().squeeze()
    with rasterio.open(mask, 'r') as f:
        mask_rect_img = f.read().squeeze()
    with rasterio.open(mask_orig, 'r') as f:
        mask_orig_img = f.read().squeeze()
    height_map = triangulation.height_map(x, y, w, h, rpc1, rpc2, H_ref, H_sec,
                                          disp_img, mask_rect_img,
                                          mask_orig_img,
                                          A=np.loadtxt(pointing))

    # write height map to a file
    common.rasterio_write(os.path.join(out_dir, 'height_map.tif'), height_map)

    if cfg['clean_intermediate']:
        common.remove(H_ref)
        common.remove(H_sec)
        common.remove(disp)
        common.remove(mask)


def disparity_to_ply(tile, i, j):
    """
    Compute a point cloud from the disparity map of a pair of image tiles.
    This function is called by s2p.main only if there are two input images (not three).
    Args:
        tile: dictionary containing the information needed to process a tile.
    """
    out_dir = os.path.join(tile['dir'], f'pair_{i}_{j}')
    ply_file = os.path.join(out_dir, 'cloud.ply')
    x, y, w, h = tile['coordinates']
    rpc1 = cfg['images'][i]['rpcm']
    rpc2 = cfg['images'][j]['rpcm']

    print('triangulating tile {} {}...'.format(x, y))
    H_ref = os.path.join(out_dir, 'H_ref.txt')
    H_sec = os.path.join(out_dir, 'H_sec.txt')
    pointing = os.path.join(cfg['out_dir'], f'global_pointing_pair_{i}_{j}.txt')
    disp = os.path.join(out_dir, 'rectified_disp.tif')

    extra = os.path.join(out_dir, 'rectified_disp_confidence.tif')
    if not os.path.exists(extra):    # confidence file not always generated
        extra = ''

    mask_rect = os.path.join(out_dir, 'rectified_mask.tif')
    mask_orig = os.path.join(out_dir, 'mask.tif')

    # prepare the image needed to colorize point cloud
    if cfg['images'][0]['clr']:
        # we want colors image and rectified_ref.tif to have the same size
        with rasterio.open(os.path.join(out_dir, 'rectified_ref.tif')) as f:
            ww, hh = f.width, f.height

        colors = common.tmpfile(".tif")
        common.image_apply_homography(colors, cfg['images'][0]['clr'], np.loadtxt(H_ref), ww, hh)
        with rasterio.open(colors, "r") as f:
            colors = f.read()

    else:
        with rasterio.open(os.path.join(out_dir, 'rectified_ref.tif')) as f:
            tile_img = f.read()
        # colors = common.linear_stretching_and_quantization_8bit(tile_img)

        ###
        # 修改：不应该按照一个tile进行拉伸，因该在整张影像的基础上线性拉伸
        ###
        with rasterio.open(cfg['images'][i]['img']) as f:
            full_img = f.read()
        colors = common.linear_stretching_and_quantization_8bit_tile(full_img, tile_img)

    # compute the point cloud
    with rasterio.open(disp, 'r') as f:
        disp_img = f.read().squeeze()
    with rasterio.open(mask_rect, 'r') as f:
        mask_rect_img = f.read().squeeze()
    with rasterio.open(mask_orig, 'r') as f:
        mask_orig_img = f.read().squeeze()

    out_crs = geographiclib.pyproj_crs(cfg['out_crs'])


    xyz_array, err = triangulation.disp_to_xyz(rpc1, rpc2,
                                               np.loadtxt(H_ref), np.loadtxt(H_sec),
                                               disp_img, mask_rect_img,
                                               img_bbx=(x, x+w, y, y+h),
                                               mask_orig=mask_orig_img,
                                               A=np.loadtxt(pointing),
                                               out_crs=out_crs)

    # 3D filtering
    r = cfg['3d_filtering_r']
    n = cfg['3d_filtering_n']
    if r and n:
        triangulation.filter_xyz(xyz_array, r, n, cfg['gsd'])

    proj_com = "CRS {}".format(cfg['out_crs'])
    triangulation.write_to_ply(ply_file, xyz_array, colors, proj_com, confidence=extra)

    if cfg['clean_intermediate']:
        common.remove(H_ref)
        common.remove(H_sec)
        common.remove(disp)
        common.remove(mask_rect)
        common.remove(mask_orig)
        common.remove(os.path.join(out_dir, 'rectified_ref.tif'))



def mean_heights(tile):
    """
    """
    w, h = tile['coordinates'][2:]
    n = len(cfg['images']) - 1
    maps = np.empty((h, w, n))
    for i in range(n):
        try:
            with rasterio.open(os.path.join(tile['dir'], 'pair_{}'.format(i + 1),
                                            'height_map.tif'), 'r') as f:
                maps[:, :, i] = f.read(1)
        except RuntimeError:  # the file is not there
            maps[:, :, i] *= np.nan

    validity_mask = maps.sum(axis=2)  # sum to propagate nan values
    validity_mask += 1 - validity_mask  # 1 on valid pixels, and nan on invalid

    # save the n mean height values to a txt file in the tile directory
    np.savetxt(os.path.join(tile['dir'], 'local_mean_heights.txt'),
               [np.nanmean(validity_mask * maps[:, :, i]) for i in range(n)])


def global_mean_heights(tiles):
    """
    """
    local_mean_heights = [np.loadtxt(os.path.join(t['dir'], 'local_mean_heights.txt'))
                          for t in tiles]
    global_mean_heights = np.nanmean(local_mean_heights, axis=0)
    for i in range(len(cfg['images']) - 1):
        np.savetxt(os.path.join(cfg['out_dir'],
                                'global_mean_height_pair_{}.txt'.format(i+1)),
                   [global_mean_heights[i]])


def heights_fusion(tile):
    """
    Merge the height maps computed for each image pair and generate a ply cloud.

    Args:
        tile: a dictionary that provides all you need to process a tile
    """
    tile_dir = tile['dir']
    height_maps = [os.path.join(tile_dir, 'pair_%d' % (i + 1), 'height_map.tif')
                   for i in range(len(cfg['images']) - 1)]

    # remove spurious matches
    if cfg['cargarse_basura']:
        for img in height_maps:
            common.cargarse_basuramean_heights(img, img)

    # load global mean heights
    global_mean_heights = []
    for i in range(len(cfg['images']) - 1):
        x = np.loadtxt(os.path.join(cfg['out_dir'],
                                    'global_mean_height_pair_{}.txt'.format(i+1)))
        global_mean_heights.append(x)

    # merge the height maps (applying mean offset to register)
    fusion.merge_n(os.path.join(tile_dir, 'height_map.tif'), height_maps,
                   global_mean_heights, averaging=cfg['fusion_operator'],
                   threshold=cfg['fusion_thresh'])

    if cfg['clean_intermediate']:
        for f in height_maps:
            common.remove(f)


def heights_to_ply(tile):
    """
    Generate a ply cloud.

    Args:
        tile: a dictionary that provides all you need to process a tile
    """
    # merge the n-1 height maps of the tile (n = nb of images)
    heights_fusion(tile)

    # compute a ply from the merged height map
    out_dir = tile['dir']
    x, y, w, h = tile['coordinates']
    plyfile = os.path.join(out_dir, 'cloud.ply')
    height_map = os.path.join(out_dir, 'height_map.tif')

    if cfg['images'][0]['clr']:
        with rasterio.open(cfg['images'][0]['clr'], "r") as f:
            colors = f.read(window=((y, y + h), (x, x + w)))
    else:
        with rasterio.open(cfg['images'][0]['img'], "r") as f:
            tile_img = f.read(window=((y, y + h), (x, x + w)))
            full_img = f.read()
        colors = common.linear_stretching_and_quantization_8bit_tile(full_img, tile_img)

        # colors = common.linear_stretching_and_quantization_8bit(colors)

    out_crs = geographiclib.pyproj_crs(cfg['out_crs'])
    xyz_array = triangulation.height_map_to_xyz(height_map,
                                                cfg['images'][0]['rpcm'], x, y,
                                                out_crs)

    # 3D filtering
    r = cfg['3d_filtering_r']
    n = cfg['3d_filtering_n']
    if r and n:
        triangulation.filter_xyz(xyz_array, r, n, cfg['gsd'])

    proj_com = "CRS {}".format(cfg['out_crs'])
    triangulation.write_to_ply(plyfile, xyz_array, colors, proj_com)

    if cfg['clean_intermediate']:
        common.remove(height_map)
        common.remove(os.path.join(out_dir, 'mask.png'))


def plys_to_dsm(tile, i, j):
    """
    Generates DSM from plyfiles (cloud.ply)

    Args:
        tile: a dictionary that provides all you need to process a tile
    """
    out_dsm = os.path.join(tile['dir'], f"pair_{i}_{j}", 'dsm.tif')
    out_conf = os.path.join(tile['dir'], f"pair_{i}_{j}", 'confidence.tif')
    r = cfg['dsm_resolution']

    # compute the point cloud x, y bounds
    points, _ = ply.read_3d_point_cloud_from_ply(os.path.join(tile['dir'], f"pair_{i}_{j}", 'cloud.ply'))
    if len(points) == 0:
        return

    xmin, ymin, *_ = np.min(points, axis=0)
    xmax, ymax, *_ = np.max(points, axis=0)

    # compute xoff, yoff, xsize, ysize on a grid of unit r
    xoff = np.floor(xmin / r) * r
    xsize = int(1 + np.floor((xmax - xoff) / r))

    yoff = np.ceil(ymax / r) * r
    ysize = int(1 - np.floor((ymin - yoff) / r))

    roi = xoff, yoff, xsize, ysize

    clouds = [os.path.join(tile['dir'], n_dir, f"pair_{i}_{j}", 'cloud.ply') for n_dir in tile['neighborhood_dirs']]
    raster, profile = plyflatten_from_plyfiles_list(clouds, resolution=r, roi=roi,
                                                    radius=cfg['dsm_radius'], sigma=cfg['dsm_sigma'])

    # save output image with utm georeferencing
    common.rasterio_write(out_dsm, raster[:, :, 0], profile=profile)

    # export confidence (optional)
    # note that the plys are assumed to contain the fields:
    # [x(float32), y(float32), z(float32), r(uint8), g(uint8), b(uint8), confidence(optional, float32)]
    # so the raster has 4 or 5 columns: [z, r, g, b, confidence (optional)]
    if raster.shape[-1] == 5:
        common.rasterio_write(out_conf, raster[:, :, 4], profile=profile)


def global_dsm(tiles):
    """
    Merge tilewise DSMs and confidence maps in a global DSM and confidence map.
    """
    # 修改：增加合并全部点云
    bounds = None
    if "roi_geojson" in cfg:
        ll_poly = geographiclib.read_lon_lat_poly_from_geojson(cfg["roi_geojson"])
        pyproj_crs = geographiclib.pyproj_crs(cfg["out_crs"])
        bounds = geographiclib.crs_bbx(ll_poly, pyproj_crs, align=cfg["dsm_resolution"])

    creation_options = {"tiled": True,
                        "blockxsize": 256,
                        "blockysize": 256,
                        "compress": "deflate",
                        "predictor": 2}

    dsms = []
    confidence_maps = []
    ply_clouds = []

    for i, j in cfg["pairs_idx"]:
        for t in tiles:
            d = os.path.join(t["dir"], f"pair_{i}_{j}", "dsm.tif")
            if os.path.exists(d):
                dsms.append(d)

            c = os.path.join(t["dir"], f"pair_{i}_{j}", "confidence.tif")
            if os.path.exists(c):
                confidence_maps.append(c)

            p = os.path.join(t["dir"], f"pair_{i}_{j}", "cloud.ply")
            if os.path.exists(p):
                ply_clouds.append(PlyData.read(p).elements[0].data)


        if dsms:
            rasterio.merge.merge(dsms,
                                bounds=bounds,
                                res=cfg["dsm_resolution"],
                                nodata=np.nan,
                                indexes=[1],
                                dst_path=os.path.join(cfg["out_dir"], f"pair_{i}_{j}_dsm.tif"),
                                dst_kwds=creation_options)

        if confidence_maps:
            rasterio.merge.merge(confidence_maps,
                                bounds=bounds,
                                res=cfg["dsm_resolution"],
                                nodata=np.nan,
                                indexes=[1],
                                dst_path=os.path.join(cfg["out_dir"], f"pair_{i}_{j}_confidence.tif"),
                                dst_kwds=creation_options)

        if ply_clouds:
            el = PlyElement.describe(np.concatenate(ply_clouds), 'vertex')
            PlyData([el]).write(os.path.join(cfg["out_dir"], f"pair_{i}_{j}_cloud.ply"))
