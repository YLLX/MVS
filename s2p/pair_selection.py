import rpcm
from config import cfg
import os
import rasterio
import sift
import numpy as np
import matplotlib.pyplot as plt
import srtm4

def get_all_pairs(img_list, return_idx=False):
    """
    计算可以组成的所有影像队

    参数:
        img_list (list): 每一个元素代表一张影像

    返回:
        list:每一个元素代表一个影像对
    """
    img_pairs = []
    img_pairs_idx = []
    for i in range(len(img_list)):
        for j in range(i + 1, len(img_list)):
            img_pairs.append((img_list[i], img_list[j]))
            img_pairs_idx.append((i, j))
    if return_idx: return img_pairs_idx
    else: return img_pairs


def get_angles(img1:str, img2:str, rpc1: rpcm.RPCModel, rpc2: rpcm.RPCModel, x, y, w, h):
    """
    计算影像对的 入射角(天顶角、方位角) 和 交会角

    参数:
        img1 (str): 主影像路径
        img2 (str): 从影像路径
        rpc1 (rpcm.RPCModel): 主影像rpc
        rpc2 (rpcm.RPCModel): 从影像rpc
        x,y,w,h: img roi
    """
    lon_center, lat_center = rpc1.localization(x + w//2, y + h//2, 0)

    ref_zenith, ref_azimuth = rpc1.incidence_angles(lon_center, lat_center, 0)
    sec_zenith, sec_azimuth = rpc2.incidence_angles(lon_center, lat_center, 0)

    ref_d = rpcm.utils.viewing_direction(ref_zenith, ref_azimuth)
    sec_d = rpcm.utils.viewing_direction(sec_zenith, sec_azimuth)

    ref_sec_angle = rpcm.angle_between_views(img1, img2, lon_center, lat_center, 0)

    return ref_zenith, ref_azimuth, sec_zenith, sec_azimuth, ref_sec_angle

def get_time(img_file):
    """
    获取每一张影像获取的时间: 返回 年，月，日

    """
    with rasterio.open(img_file) as src:
        date_time = src.tags()['NITF_IDATIM']
        year = int(date_time[0:4])
        month = int(date_time[4:6])
        day = int(date_time[6:8])
    return year, month, day


def select_pairs(img_list, x, y, w, h, return_idx=False):
    """
    输入:
        img_list: 影像列表
        w, h: 影像宽高
    返回:
        list: 选择后的影像对
    """
    selected_pairs = []
    select_pairs_idx = []
    selected_angles = []
    month_dists = []
    sift_matches = []

    for ref_idx, sec_idx in get_all_pairs(img_list, return_idx=True):
        img_ref = img_list[ref_idx]
        img_sec = img_list[sec_idx]
        # 天顶角 < 40 ; 5 < 交会角 < 45
        ref_zenith, _, sec_zenith, _, ref_sec_angle = get_angles(img_ref["img"], img_sec["img"],
                                                                 img_ref["rpcm"], img_sec["rpcm"],
                                                                 x, y, w, h)
        if (ref_zenith > 40) or (sec_zenith > 40) or (ref_sec_angle > 45) or (ref_sec_angle < 5):
            continue

        # 月间隔 < 1
        _, ref_month, _ = get_time(img_ref["img"])
        _, sec_month, _ = get_time(img_sec["img"])
        dist = abs(ref_month - sec_month)
        if min(dist, 12 - dist) >= 1:
            continue
        # sift matches
        m = sift.matches_on_rpc_roi(img_ref["img"], img_sec["img"],
                                    img_ref["rpcm"], img_sec["rpcm"],
                                    x, y, w, h, 'relative',
                                    cfg['sift_match_thresh'], cfg['max_pointing_error'])
        if m is None: continue
        if  m.shape[0] < 50: continue

        selected_pairs.append((img_ref, img_sec))
        select_pairs_idx.append((ref_idx, sec_idx))
        selected_angles.append((ref_zenith, sec_zenith, ref_sec_angle))
        month_dists.append(dist)
        sift_matches.append(m.shape[0])

    # 按照 sift 排序
    sift_matches, selected_pairs, select_pairs_idx = [list(x) for x in
        zip(*sorted(zip(sift_matches, selected_pairs, select_pairs_idx), key=lambda x:x[0]))]
    # save
    with open(os.path.join(cfg["out_dir"], "selected_pairs.txt"), 'w') as f:
        for ref_idx, sec_idx in select_pairs_idx:
            f.write(f"{ref_idx} {sec_idx}\n")

    if return_idx:
        return select_pairs_idx
    else:
        return selected_pairs



def get_closest_pairs(img_list, x, y, w, h):
    # 按照方位角获取 circle pair
    n = len(img_list)
    azimuth_list = []
    zenith_list = []
    direction_list = []
    for img in img_list:
        rpc = img['rpcm']
        lon, lat = rpc.localization(x + w//2, y + h//2, 0)
        z = srtm4.srtm4(lon, lat)

        zenith, azimuth = rpc.incidence_angles(lon, lat, z)
        satellite_direction = rpcm.utils.viewing_direction(zenith, azimuth)

        azimuth_list.append(azimuth)
        zenith_list.append(zenith)
        direction_list.append(satellite_direction)

    # 按照 方位角（azimuth） 排序
    index = np.array(sorted(range(n), key=lambda k: azimuth_list[k], reverse=True))

    # 过滤 不符合要求的影像
    for i in index:  # 不满足要求的影像 设置为 -1
        if zenith_list[i] > 40:
            index[i] = -1
    # 获取 pairs, 按照方位角大小顺序
    pairs_list = []
    for i in range(n):
        ref_idx = index[i]
        sec_idx = index[(i+1) % n]
        ref_img = img_list[ref_idx]['img']
        sec_img = img_list[sec_idx]['img']
        # 过滤影像对，包括 交会角 和 时间差
        ref_sec_angle = rpcm.angle_between_views(ref_img, sec_img, lon, lat, z)
        if (ref_sec_angle > 45) or (ref_sec_angle < 5):
            continue
        _, ref_month, ref_day = get_time(ref_img)
        _, sec_month, sec_day = get_time(sec_img)
        month_dist = abs(ref_month - sec_month)
        if min(month_dist, 12 - month_dist) > 1:
            continue
        # day_diff = 15
        # if min(month_dist, 12 - month_dist) == 0 and abs(ref_day - sec_day) >= day_diff:
        #     continue
        # if min(month_dist, 12 - month_dist) == 1 and (30 - ref_day + sec_day) >= day_diff:
        #     continue
        pairs_list.append((ref_idx, sec_idx))
        # print(azimuth_list[ref_idx], azimuth_list[sec_idx])

    # 绘制卫星方向图
    if False:
        directions = np.array(direction_list)
        u = directions[index!=-1, 0]
        v = directions[index!=-1, 1]
        w = directions[index!=-1, 2]

        x = np.zeros_like(u)
        y = np.zeros_like(u)
        z = np.zeros_like(u)

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        ax1.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
        ax2.scatter(u, v)

        plt.show()

    return pairs_list



def initialize_centroids(points, k):
    """returns k centroid from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    dist = np.zeros((len(points), len(centroids)))
    for i, c in enumerate(centroids):
        diff = np.abs(c - points)
        diff[diff > 180] = 360 - diff[diff > 180]
        dist[:, i] = diff
    return np.argmin(dist, axis=1)

def move_centroids(points, closest, k):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([np.mean(points[closest == i]) for i in range(k)])


def group_by_cluster(img_list, x, y, w, h, max_k=12):

    # 获取每张影像的方位角 和 摄影时间
    azimuth_list = []
    time_list = []
    for img in img_list:
        rpc = img['rpcm']
        lon, lat = rpc.localization(x + w//2, y + h//2, 0)
        _, azimuth = rpc.incidence_angles(lon, lat, 0)
        _, month, day = get_time(img['img'])
        azimuth_list.append(azimuth)
        time_list.append(month * 30 + day)

    # k-median clustering
    points = np.array(azimuth_list)
    assert(max_k >= 4)
    for k in range(4, max_k+1):
        init_centroids = initialize_centroids(points, k)
        while True:
            closest = closest_centroid(points, init_centroids)
            centroids = move_centroids(points, closest, k)
            if np.all( init_centroids - centroids == 0 ): break
            else: init_centroids = centroids
        # 判断聚类是否正确
        # To Do:
        cluster_list = [np.where(closest == i)[0] for i in range(k)]
        std_list = [int(np.std(points[c_idx])) for c_idx in cluster_list]
        if np.all(np.array(std_list) <= 15):
            break
    # 至少两张影像为一个类
    cluster_list = [cluster for cluster in cluster_list if len(cluster)>1]
    # 选择主影像,选择日期中值
    ref_list = [cluster[np.argsort(np.array(time_list)[cluster])[len(cluster) // 2]]
                for cluster in cluster_list]

    # 可视化
    if False:
        fig, ax = plt.subplots()
        for cluster in cluster_list:
            data = points[cluster]
            # print(data)
            x = np.cos(data*np.pi/180.0)
            y = np.sin(data*np.pi/180.0)
            ax.scatter(x, y)
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
        ax.set_aspect(1)
        plt.show()

    return ref_list, cluster_list


def get_tri_pairs(img_list, ref_list, cluster_list, x, y, w, h):
    group_pair_list = []
    for ref, cluster in zip(ref_list, cluster_list):
        best_pairs = []
        for c in cluster:
            if c == ref: continue
            img1 = img_list[c]
            _, f_month, f_day = get_time(img1['img'])
            lon, lat = img1['rpcm'].localization(x + w//2, y + h//2, 0)
            z = srtm4.srtm4(lon, lat)

            max_angle = 0
            max_idx = 0
            for i, img2 in enumerate(img_list):
                _, s_month, s_day = get_time(img2['img'])
                dt = abs(s_month*12 + s_day - (f_month*12 + f_day))
                dt = dt if dt<12*30 else 12*30-dt
                if dt < 30:
                    angle = rpcm.angle_between_views(img1['img'], img2['img'], lon, lat, z)
                    if angle > max_angle: max_idx = i
            best_pairs.append((ref, c, max_idx))
        group_pair_list.append(best_pairs)
    return group_pair_list

def draw_pairs(img_list, pairs, x, y, w, h):
    directions = []
    for img in img_list:
        rpc = img['rpcm']
        lon, lat = rpc.localization(x + w//2, y + h//2, 0)
        zenith, azimuth = rpc.incidence_angles(lon, lat, 0)
        directions.append(rpcm.utils.viewing_direction(zenith, azimuth))
    dx = np.array(directions)[:, 0]
    dy = np.array(directions)[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(dx, dy)
    for cluster_pair in pairs:
        for pair in cluster_pair:
            ax.plot([dx[pair[0]], dx[pair[1]]], [dy[pair[0]], dy[pair[1]]], 'green')
            ax.plot([dx[pair[1]], dx[pair[2]]], [dy[pair[1]], dy[pair[2]]], 'red')
            ax.plot([dx[pair[0]], dx[pair[2]]], [dy[pair[0]], dy[pair[2]]], 'blue')
    ax.set_aspect(1)
    plt.show()


if __name__ == "__main__":
    img_dir = "./data/input/JAX_167/images"
    img_list = []
    dir_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
    for i in range(len(dir_list)):
        img_list.append({"img": dir_list[i], "rpcm": rpcm.rpc_from_geotiff(dir_list[i])})

    # pairs_list = get_closest_pairs(img_list, 0, 0, 2048, 2048)
    ref, cluster = group_by_cluster(img_list, 0, 0, 2048, 2048)
    tri_pairs = get_tri_pairs(img_list, ref, cluster, 0, 0, 2048, 2048)
    draw_pairs(img_list, tri_pairs, 0, 0, 2048, 2048)