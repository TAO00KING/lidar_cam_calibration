#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import math
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import threading

class LiveProjection:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 最新数据
        self.latest_image = None
        self.latest_pointcloud = None
        self.image_received = False
        self.pc_received = False
        
        # 相机内参 (使用你的标定结果)
        self.cx = 917.032444      # 根据标定结果修改
        self.cy = 497.956248      # 根据标定结果修改
        self.fx = 685.99056       # 根据标定结果修改
        self.fy = 685.29179       # 根据标定结果修改
        self.img_width = 1920     # 根据实际图像宽度修改
        self.img_height = 1080    # 根据实际图像高度修改
        
        # ========== 选择一组外参（根据你的标定结果） ==========
        # 选项1: Ceres 3d3d 优化结果
        self.set_extrinsic(0.0412387, 0.0902474, 0.0706924, 
                          112.185, -88.5105, -22.554)
        
        # 选项2: Ceres-q_t-3d2d 优化结果（推荐，重投影误差最小）
        # self.set_extrinsic(0.0516832, 0.109913, 0.0318952,
        #                   91.2812, -0.786187, 89.2826)
        
        # 选项3: SVD 3d3d 结果
        # self.set_extrinsic(0.0412418, 0.0902436, 0.0706909,
        #                   112.174, -88.5103, -22.5435)
        
        # 选项4: 如果上面的都不对，可以尝试这个（从相机到激光雷达的转换）
        # self.set_extrinsic(-0.06135, 1.5056, 0.22248,
        #                   167.161, -174.17, 1.01688)
        
        # 投影参数
        self.max_range = 30.0     # 最大投影距离(米)
        self.min_range = 0.1      # 最小投影距离(米)
        self.downsample = 1       # 点云降采样（1表示不降采样）
        
        # 订阅话题（根据你的实际话题名称修改）
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.pc_sub = rospy.Subscriber("/lidar_points_synced", PointCloud2, self.pc_callback, queue_size=1)
        
        # 发布投影图像
        self.result_pub = rospy.Publisher("/projection_result", Image, queue_size=1)
        
        # 启动定时器，定期处理
        self.timer = rospy.Timer(rospy.Duration(0.033), self.process_callback)  # 30Hz
        
        print("="*60)
        print("实时点云投影节点启动")
        print("="*60)
        print("相机内参:")
        print(f"  fx={self.fx}, fy={self.fy}")
        print(f"  cx={self.cx}, cy={self.cy}")
        print(f"  图像尺寸: {self.img_width}x{self.img_height}")
        print("="*60)
        print("订阅话题:")
        print("  图像: /usb_cam/image_raw")
        print("  点云: /lidar_points_synced")
        print("发布话题:")
        print("  结果: /projection_result")
        print("="*60)
        
    def set_extrinsic(self, tx, ty, tz, yaw_deg, pitch_deg, roll_deg):
        """设置外参"""
        # 平移
        self.t = np.array([tx, ty, tz])
        
        # 欧拉角转弧度
        yaw = yaw_deg * math.pi / 180.0
        pitch = pitch_deg * math.pi / 180.0
        roll = roll_deg * math.pi / 180.0
        
        # 旋转矩阵 (ZYX顺序)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
        
        self.R = Rz @ Ry @ Rx
        
        print("\n外参设置:")
        print(f"  平移: ({tx:.4f}, {ty:.4f}, {tz:.4f})")
        print(f"  欧拉角: ({yaw_deg:.2f}, {pitch_deg:.2f}, {roll_deg:.2f})")
        print(f"  旋转矩阵:\n{self.R}")
        
    def image_callback(self, msg):
        """图像回调"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 更新图像尺寸
            if self.latest_image is not None:
                h, w = self.latest_image.shape[:2]
                if w != self.img_width or h != self.img_height:
                    self.img_width = w
                    self.img_height = h
                    print(f"更新图像尺寸: {self.img_width}x{self.img_height}")
            self.image_received = True
        except Exception as e:
            rospy.logerr(f"图像转换失败: {e}")
    
    def pc_callback(self, msg):
        """点云回调"""
        self.latest_pointcloud = msg
        self.pc_received = True
    
    def get_color_by_distance(self, distance):
        """根据距离返回鲜艳的颜色 (BGR格式)"""
        # 定义距离区间和对应的鲜艳颜色
        if distance < 2.0:
            # 极近: 亮红色
            return (0, 0, 255)
        elif distance < 5.0:
            # 近: 橙色 (BGR: 0, 165, 255)
            return (0, 165, 255)
        elif distance < 10.0:
            # 中近: 黄色 (BGR: 0, 255, 255)
            return (0, 255, 255)
        elif distance < 15.0:
            # 中等: 亮绿色 (BGR: 0, 255, 0)
            return (0, 255, 0)
        elif distance < 20.0:
            # 中远: 青色 (BGR: 255, 255, 0)
            return (255, 255, 0)
        elif distance < 25.0:
            # 远: 亮蓝色 (BGR: 255, 0, 0)
            return (255, 0, 0)
        else:
            # 极远: 紫色 (BGR: 255, 0, 255)
            return (255, 0, 255)
    
    def get_color_gradient(self, distance):
        """使用渐变色，使颜色过渡更平滑"""
        # 归一化距离 (0-30米映射到0-1)
        normalized = min(distance / 30.0, 1.0)
        
        # 使用HSV色彩空间转换，获得更丰富的颜色
        # 色相: 从红色(0) 到 紫色(300)，经过黄、绿、青、蓝
        hue = int(300 * normalized)  # 0=红, 60=黄, 120=绿, 180=青, 240=蓝, 300=紫
        saturation = 255  # 全饱和
        value = 255       # 全亮度
        
        # HSV 转 BGR
        hsv = np.uint8([[[hue, saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return (int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2]))
    
    def project_pointcloud(self, img, pc_msg):
        """将点云投影到图像上"""
        if img is None:
            return None
        
        img_result = img.copy()
        
        # 解析点云
        try:
            # 获取点云数据
            points = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        except Exception as e:
            rospy.logerr(f"点云解析失败: {e}")
            cv2.putText(img_result, f"Point cloud parse error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return img_result
        
        if len(points) == 0:
            cv2.putText(img_result, "No point cloud data", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return img_result
        
        # 降采样
        step = self.downsample
        points = points[::step]
        
        projected_count = 0
        behind_count = 0
        out_count = 0
        
        # 统计各个距离区间的点数
        distance_stats = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        
        for pt in points:
            x, y, z = pt
            
            # 距离过滤
            dist = math.sqrt(x*x + y*y + z*z)
            if dist < self.min_range or dist > self.max_range:
                continue
            
            # 激光雷达到相机坐标系
            pt_cam = self.R @ np.array([x, y, z]) + self.t
            
            # 检查是否在相机前方
            if pt_cam[2] <= 0.1:
                behind_count += 1
                continue
            
            # 投影到像素坐标
            u = int(self.fx * pt_cam[0] / pt_cam[2] + self.cx)
            v = int(self.fy * pt_cam[1] / pt_cam[2] + self.cy)
            
            # 检查是否在图像范围内
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                projected_count += 1
                
                # 使用鲜艳的颜色（二选一）
                # 方案1: 区间颜色（更鲜明的区分）
                color = self.get_color_by_distance(dist)
                
                # 方案2: 渐变色（平滑过渡，注释掉上面的，取消注释下面的）
                # color = self.get_color_gradient(dist)
                
                # 统计距离区间
                if dist < 2.0:
                    distance_stats[0] += 1
                elif dist < 5.0:
                    distance_stats[1] += 1
                elif dist < 10.0:
                    distance_stats[2] += 1
                elif dist < 15.0:
                    distance_stats[3] += 1
                elif dist < 20.0:
                    distance_stats[4] += 1
                elif dist < 25.0:
                    distance_stats[5] += 1
                else:
                    distance_stats[6] += 1
                
                # 绘制点（增大点的大小使其更明显）
                # 根据距离调整点大小：近的点大，远的点小
                if dist < 5.0:
                    point_size = 3
                elif dist < 15.0:
                    point_size = 2
                else:
                    point_size = 1
                
                cv2.circle(img_result, (u, v), point_size, color, -1)
            else:
                out_count += 1
        
        # 添加统计信息（背景半透明）
        overlay = img_result.copy()
        cv2.rectangle(overlay, (5, 5), (350, 200), (0, 0, 0), -1)
        img_result = cv2.addWeighted(overlay, 0.6, img_result, 0.4, 0)
        
        info_y = 30
        cv2.putText(img_result, f"Projected Points: {projected_count}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_result, f"Total Points: {len(points)} (Downsample: {self.downsample})", (10, info_y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img_result, f"Behind Camera: {behind_count}, Out of Frame: {out_count}", (10, info_y+55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 距离统计
        cv2.putText(img_result, "Distance Distribution:", (10, info_y+85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset = info_y + 105
        intervals = ["<2m", "2-5m", "5-10m", "10-15m", "15-20m", "20-25m", ">25m"]
        colors = [(0,0,255), (0,165,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0), (255,0,255)]
        for i, (interval, color) in enumerate(zip(intervals, colors)):
            cv2.circle(img_result, (15, y_offset + i*20), 4, color, -1)
            cv2.putText(img_result, f"{interval}: {distance_stats[i]}", (30, y_offset + i*20 + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 显示当前使用的外参
        cv2.putText(img_result, f"Extrinsic - T: ({self.t[0]:.3f}, {self.t[1]:.3f}, {self.t[2]:.3f})", 
                   (10, self.img_height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 彩色图例（底部居中）
        legend_width = 350
        legend_height = 30
        legend_x = (self.img_width - legend_width) // 2
        legend_y = self.img_height - 30
        
        # 创建渐变图例
        for i in range(legend_width):
            dist = i / legend_width * 30.0  # 0-30米
            color = self.get_color_by_distance(dist)
            cv2.line(img_result, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 1)
        
        # 图例标注
        cv2.putText(img_result, "0m", (legend_x, legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_result, "5m", (legend_x + int(legend_width * 5/30), legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_result, "10m", (legend_x + int(legend_width * 10/30), legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_result, "15m", (legend_x + int(legend_width * 15/30), legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_result, "20m", (legend_x + int(legend_width * 20/30), legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_result, "25m", (legend_x + int(legend_width * 25/30), legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_result, "30m", (legend_x + legend_width - 20, legend_y + legend_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img_result
    
    def process_callback(self, event):
        """定时处理回调"""
        if not self.image_received or not self.pc_received:
            return
        
        # 投影
        result_img = self.project_pointcloud(self.latest_image, self.latest_pointcloud)
        
        if result_img is not None:
            # 发布结果
            result_msg = self.bridge.cv2_to_imgmsg(result_img, "bgr8")
            result_msg.header.stamp = rospy.Time.now()
            self.result_pub.publish(result_msg)
            
            # 显示 (可选)
            cv2.imshow("Live Projection", result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User quit")

def main():
    rospy.init_node("live_projection", anonymous=True)
    projector = LiveProjection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\n节点关闭")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
