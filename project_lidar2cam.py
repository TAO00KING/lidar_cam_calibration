#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云投影到图像验证脚本
==================

功能：
  读取 PCD 点云和图像，用标定外参将雷达点投影到图像平面，
  投影效果越好说明标定结果越准确。

使用方法：
  # 方式1：从 result.txt 自动读取外参
  python3 project_lidar2cam.py \
      --pcd data/cloud/xxx.pcd \
      --image data/image/xxx.png \
      --result data/result.txt \
      --method Ceres_3d3d

  # 方式2：直接指定外参（欧拉角 xyz 顺序，单位度）
  python3 project_lidar2cam.py \
      --pcd data/cloud/xxx.pcd \
      --image data/image/xxx.png \
      --r 112.185 -88.5105 -22.554 \
      --t 0.04124 0.09025 0.07069

  # 方式3：ROS 实时投影（需要 ROS 环境）
  python3 project_lidar2cam.py --ros

注意：
  result.txt 中的欧拉角是 Eigen 的 eulerAngles(0,1,2) 顺序，
  即 (rx, ry, rz)，对应旋转矩阵 R = Rz * Ry * Rx。
"""

import argparse
import numpy as np
import cv2
import os
import sys

# ============================================================
#  PCD 文件读取
# ============================================================

def load_pcd_ascii(pcd_path):
    """
    读取 ASCII 格式的 PCD 文件。
    返回 Nx3 的 numpy 数组 (x, y, z)，坐标系：x前 y左 z上。
    """
    points = []
    start = 0
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('POINTS') or line.startswith('points'):
            pass  # 可以读点数，但不是必须
        if line == 'DATA ascii' or line == 'data ascii':
            start = i + 1
            break
    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                points.append([x, y, z])
            except ValueError:
                pass
    print(f"[PCD] 读取 {len(points)} 个点")
    return np.array(points, dtype=np.float64)


# ============================================================
#  欧拉角 <-> 旋转矩阵（与 Eigen eulerAngles(0,1,2) 一致）
# ============================================================

def euler_to_rotmat(rx_deg, ry_deg, rz_deg):
    """
    Eigen eulerAngles(0,1,2) 对应内旋 (intrinsic) XYZ：
      R = Rz * Ry * Rx
    输入：rx, ry, rz（度），输出：3x3 旋转矩阵
    """
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = np.array([[1,           0,            0],
                    [0,  np.cos(rx), -np.sin(rx)],
                    [0,  np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                    [          0, 1,          0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz),  np.cos(rz), 0],
                    [         0,           0, 1]])
    return Rz @ Ry @ Rx


# ============================================================
#  解析 result.txt
# ============================================================

def parse_result_txt(result_path):
    """
    从 result.txt 解析各方法的外参。
    返回 dict: {method: {"euler": [rx,ry,rz], "t": [tx,ty,tz]}}
    """
    results = {}
    current = None
    buf = []

    with open(result_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                # 识别方法段
                if 'SVD-3d3d' in line:
                    current = 'SVD_3d3d'
                    results[current] = {}
                elif 'Ceres-3d3d' in line and 'ceres' not in line.replace('Ceres-3d3d',''):
                    current = 'Ceres_3d3d'
                    results[current] = {}
                elif 'PNP' in line or 'PnP' in line:
                    current = 'PnP_3d2d'
                    results[current] = {}
                elif 'ceres-q_t' in line.lower():
                    if current is None or '3d3d' in current:
                        current = 'Ceres_ICP_3d3d'
                    else:
                        current = 'Ceres_3d2d'
                    results[current] = {}
                # 读取数值行
                if buf and current:
                    try:
                        vals = list(map(float, buf))
                        if len(vals) == 3 and 'euler' not in results[current]:
                            results[current]['euler'] = vals
                        elif len(vals) == 3 and 't' not in results[current]:
                            results[current]['t'] = vals
                    except ValueError:
                        pass
                    buf = []
                continue

            if current is None:
                continue

            # 累加数值行（可能一行或三行）
            try:
                nums = list(map(float, line.replace('[','').replace(']','').split(',')))
                buf.extend(nums)
                if len(buf) >= 3:
                    if 'euler' not in results[current]:
                        results[current]['euler'] = buf[:3]
                        buf = buf[3:]
                    elif 't' not in results[current]:
                        results[current]['t'] = buf[:3]
                        buf = []
            except ValueError:
                pass

    # 文件末尾
    if buf and current:
        try:
            vals = list(map(float, buf))
            if len(vals) == 3 and 't' not in results[current]:
                results[current]['t'] = vals
        except ValueError:
            pass

    return results


# ============================================================
#  投影核心函数
# ============================================================

def project_lidar_to_image(points_lidar, R_cl, t_cl, K, dist_coeffs, img_shape):
    """
    将雷达点云投影到图像平面。

    坐标系说明（ROS 常规）：
      雷达：x前 y左 z上
      相机：x右 y下 z前
      P_cam = R_cl @ P_lidar + t_cl

    参数：
      points_lidar : Nx3 ndarray
      R_cl          : 3x3 旋转矩阵（雷达→相机）
      t_cl          : 3   平移向量（雷达→相机）
      K             : 3x3 内参矩阵
      dist_coeffs   : (5,) 畸变系数 [k1,k2,p1,p2,k3]
      img_shape     : (h, w)

    返回：
      uvs   : Nx2 像素坐标
      depths: N    相机坐标系 Z（深度）
      mask  : N    bool，是否在前方且落在图像内
    """
    # 变换到相机坐标系
    points_cam = (R_cl @ points_lidar.T).T + t_cl.reshape(1, 3)
    depths = points_cam[:, 2]

    # 过滤：相机前方的点
    front = depths > 0.2

    # 投影：u = fx*X/Z + cx,  v = fy*Y/Z + cy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    eps = 1e-6
    inv_z = 1.0 / np.maximum(depths, eps)
    u = fx * points_cam[:, 0] * inv_z + cx
    v = fy * points_cam[:, 1] * inv_z + cy

    # 畸变校正（可选，对像素坐标做）
    # 这里简化：如果畸变较小，可以直接用；否则需要对 3D 点做畸变再投影
    # 当前用 cv2.undistortPoints 对 uv 做校正：
    if abs(dist_coeffs[0]) > 1e-6 or abs(dist_coeffs[1]) > 1e-6:
        uv_norm = np.column_stack([(u - cx)/fx, (v - cy)/fy])
        uv_norm = uv_norm.reshape(-1, 1, 2)
        uv_undist = cv2.undistortPoints(uv_norm, K, dist_coeffs, None, K)
        u = uv_undist.reshape(-1, 2)[:, 0]
        v = uv_undist.reshape(-1, 2)[:, 1]

    h, w = img_shape[:2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = front & in_img

    uvs = np.column_stack([u, v])
    return uvs, depths, mask


# ============================================================
#  绘制
# ============================================================

def draw_projection(img, uvs, depths, mask, max_depth=30.0):
    """
    在图像上绘制投影点，按深度着色：近=暖色，远=冷色。
    """
    out = img.copy()
    uvs_m = uvs[mask].astype(np.int32)
    d_m   = depths[mask]
    d_norm = np.clip(d_m / max_depth, 0.0, 1.0)

    # 对大量点做降采样绘制（加速）
    if len(uvs_m) > 50000:
        idx = np.random.choice(len(uvs_m), 50000, replace=False)
        uvs_m = uvs_m[idx]
        d_norm = d_norm[idx]

    for (u, v), dn in zip(uvs_m, d_norm):
        # 色调映射：dn=0(近)→红色，dn=1(远)→蓝色
        b = int(255 * (1.0 - dn))
        r = int(255 * dn)
        g = int(255 * (1.0 - 2.0*abs(dn - 0.5)))
        g = np.clip(g, 0, 255)
        cv2.circle(out, (int(u), int(v)), 2, (b, g, r), -1)

    return out


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="点云→相机投影验证标定结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
                "  python3 %(prog)s --pcd data/cloud/xxx.pcd --image data/image/xxx.png --result data/result.txt\n"
                "  python3 %(prog)s --pcd xxx.pcd --image xxx.png --r 112.2 -88.5 -22.5 --t 0.041 0.090 0.071"
    )
    parser.add_argument('--pcd',    type=str, help='PCD 文件路径')
    parser.add_argument('--image',  type=str, help='图像文件路径 (png/jpg)')
    parser.add_argument('--result', type=str, help='result.txt 路径')
    parser.add_argument('--method', type=str, default='Ceres_3d3d',
                        choices=['SVD_3d3d', 'Ceres_3d3d', 'PnP_3d2d',
                                 'Ceres_ICP_3d3d', 'Ceres_3d2d'],
                        help='result.txt 中用哪种方法 (默认 Ceres_3d3d)')
    # 内外参直接指定
    parser.add_argument('--fx', type=float, default=685.990560)
    parser.add_argument('--fy', type=float, default=685.291790)
    parser.add_argument('--cx', type=float, default=917.032444)
    parser.add_argument('--cy', type=float, default=497.956248)
    parser.add_argument('--k1', type=float, default=0.006154)
    parser.add_argument('--k2', type=float, default=-0.021981)
    parser.add_argument('--p1', type=float, default=-0.001910)
    parser.add_argument('--p2', type=float, default=0.001129)
    parser.add_argument('--r',  type=float, nargs=3,
                        help='欧拉角 rx ry rz（度，xyz顺序）')
    parser.add_argument('--t',  type=float, nargs=3,
                        help='平移 tx ty tz（米）')
    parser.add_argument('--max_depth', type=float, default=30.0,
                        help='显示最大深度（米），默认30')
    args = parser.parse_args()

    # ---- 1. 加载图像 ----
    if args.image and os.path.isfile(args.image):
        image = cv2.imread(args.image)
        if image is None:
            print(f"[Error] 无法读取图像: {args.image}")
            return
        print(f"[Image] {args.image}  {image.shape[1]}x{image.shape[0]}")
    else:
        print("[Error] 必须通过 --image 指定图像文件")
        return

    # ---- 2. 加载点云 ----
    if not (args.pcd and os.path.isfile(args.pcd)):
        print("[Error] 必须通过 --pcd 指定 PCD 文件")
        return
    points_lidar = load_pcd_ascii(args.pcd)
    if len(points_lidar) == 0:
        print("[Error] PCD 文件为空")
        return
    print(f"[PCD]   {len(points_lidar)} 个点")

    # ---- 3. 内外参 ----
    K = np.array([[args.fx, 0,       args.cx],
                   [0,       args.fy, args.cy],
                   [0,       0,       1    ]], dtype=np.float64)
    dist = np.array([args.k1, args.k2, args.p1, args.p2, 0.0], dtype=np.float64)

    if args.r and args.t:
        rx, ry, rz = args.r
        t_cl = np.array(args.t, dtype=np.float64)
        R_cl = euler_to_rotmat(rx, ry, rz)
        method_name = "Manual"
    elif args.result:
        res = parse_result_txt(args.result)
        if args.method not in res:
            print(f"[Error] {args.method} 不在 result.txt 中")
            print(f"        可用: {list(res.keys())}")
            return
        if 'euler' not in res[args.method] or 't' not in res[args.method]:
            print(f"[Error] {args.method} 数据不完整")
            return
        rx, ry, rz = res[args.method]['euler']
        t_cl = np.array(res[args.method]['t'], dtype=np.float64)
        R_cl = euler_to_rotmat(rx, ry, rz)
        method_name = args.method
    else:
        print("[Error] 必须通过 --r/--t 或 --result 指定外参")
        return

    print(f"\n[外参] 方法: {method_name}")
    print(f"  欧拉角 rx,ry,rz (度) = {rx:.3f}, {ry:.3f}, {rz:.3f}")
    print(f"  平移 t (米)          = {t_cl[0]:.4f}, {t_cl[1]:.4f}, {t_cl[2]:.4f}")
    print(f"  旋转矩阵 R_cl:")
    for row in R_cl:
        print(f"    {row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}")

    # ---- 4. 投影 ----
    uvs, depths, mask = project_lidar_to_image(
        points_lidar, R_cl, t_cl, K, dist, image.shape
    )
    n_total  = len(points_lidar)
    n_front  = int(np.sum(depths > 0.2))
    n_proj   = int(np.sum(mask))
    print(f"\n[投影] 总点数:    {n_total}")
    print(f"[投影] 相机前方:  {n_front}")
    print(f"[投影] 图像内:    {n_proj}")

    if n_proj == 0:
        print("[Warn] 没有点投影到图像内，请检查外参！")
        return

    # ---- 5. 绘制 ----
    result = draw_projection(image, uvs, depths, mask, max_depth=args.max_depth)

    # 标注信息
    info  = f"Method: {method_name}"
    info2 = f"t = [{t_cl[0]:.3f}, {t_cl[1]:.3f}, {t_cl[2]:.3f}]"
    info3 = f"Projected: {n_proj}/{n_total}"
    cv2.putText(result, info,  (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(result, info2, (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(result, info3, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ---- 6. 显示 ----
    h, w = result.shape[:2]
    if w > 1800:
        scale = 1800.0 / w
        display = cv2.resize(result, None, fx=scale, fy=scale)
    else:
        display = result

    cv2.imshow("Lidar Projection — 按 s 保存，任意键退出", display)
    print("\n[操作] 按 's' 保存到文件，按其他键退出")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        out = "projection_result.jpg"
        cv2.imwrite(out, result)
        print(f"[Save] 已保存: {out}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
