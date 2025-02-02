# -*- coding: utf-8 -*-
"""
Image Distortion Correction Demo

This script demonstrates the polynomial-based image distortion correction system.
It includes:
- Grid image generation for testing
- Distortion simulation with noise
- Parameter estimation from control points
- Image correction using polynomial transformation
- Result visualization and comparison

Features:
1. Grid Generation
   - Configurable grid density and size
   - Anti-aliasing with Gaussian blur

2. Distortion Simulation
   - Polynomial-based distortion transform
   - Optional Gaussian noise

3. Parameter Estimation
   - Random sampling of feature points
   - Least squares coefficient estimation
   - Automatic comparison with ground truth

4. Image Correction
   - Inverse polynomial transformation
   - High-precision bilinear interpolation

5. Visualization
   - Multi-window comparison
   - RGB channel overlay option

图像畸变校正演示

本脚本演示基于多项式的图像畸变校正系统。
包含功能：
- 用于测试的网格图像生成
- 带噪声的畸变模拟
- 从控制点估计参数
- 使用多项式变换的图像校正
- 结果可视化和对比

特性：
1. 网格生成
   - 可配置的网格密度和尺寸
   - 使用高斯模糊的抗锯齿处理

2. 畸变模拟
   - 基于多项式的畸变变换
   - 可选的高斯噪声

3. 参数估计
   - 特征点随机采样
   - 最小二乘系数估计
   - 与真实值自动对比

4. 图像校正
   - 逆多项式变换
   - 高精度双线性插值

5. 可视化
   - 多窗口对比
   - RGB通道叠加选项
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage
import cv2
from matrix_polynomial_math import generate_powers, compute_polynomial, backward_transform
import time
import threading
from typing import Tuple, Optional

# ===================== 全局可调参数 =====================
INTERP_RANGE = 4         # 插值邻域范围，出现细碎空区时增大该值

# 畸变模型参数
POLY_COEFFS_A = [1,  0e-3, -1e-1, 0e-1, -1e-2, 0e-9]  # 多项式A系数（各阶系数）
POLY_COEFFS_B = [0, -1e-1, -0e-2, 1e-1,  0e-6, 1e-4]  # 多项式B系数（各阶系数）
I_MATRIX = np.eye(2)     # 单位矩阵（用于生成线性项）
G_MATRIX = np.array([[0, 1], [1, 0]])  # 交换矩阵（用于生成交叉项）

# 高斯模糊参数
GAUSSIAN_KERNEL = (3, 3) # 高斯核尺寸（宽, 高，需为奇数）
GAUSSIAN_SIGMA = 0.3     # 高斯分布标准差（像素）

# 特征点参数
SAMPLE_POINTS = 50       # 用于求解的采样点数量（实际采样数不超过像素总数）
NOISE_STD = 2.0          # 高斯噪声标准差（单位：像素）

# 网格参数
GRID_COLS = 32           # 网格横向单元格数量（X方向）
GRID_ROWS = 24           # 网格纵向单元格数量（Y方向）
CELL_SIZE = 50           # 单元格边长（像素）
LINE_WIDTH = 1           # 网格线宽度（像素）

# 显示参数
FIGURE_SIZE = (12, 8)    # 显示窗口物理尺寸（宽, 高，单位：英寸）
DISPLAY_DPI = 300         # 图像显示分辨率（每英寸点数）
FONT_FAMILY = 'Microsoft YaHei'  # 字体名称（需系统支持）
FONT_SIZE = 8            # 字体大小（磅）
FONT_COLOR = 'black'     # 标题字体颜色

# ===================== 核心功能实现 =====================
def create_grid_image():
    """
    生成带高斯模糊的标准网格图像
    返回：
        高斯模糊处理后的网格图像（uint8格式）
    """
    # 初始化白色背景图像
    image = np.ones((GRID_ROWS*CELL_SIZE, GRID_COLS*CELL_SIZE), dtype=np.uint8)*255
    
    # 绘制水平网格线
    for i in range(GRID_ROWS + 1):
        y = i * CELL_SIZE
        start = max(0, y - LINE_WIDTH//2)
        end = min(image.shape[0], y + LINE_WIDTH//2 + 1)
        image[start:end, :] = 0
    
    # 绘制垂直网格线
    for j in range(GRID_COLS + 1):
        x = j * CELL_SIZE
        start = max(0, x - LINE_WIDTH//2)
        end = min(image.shape[1], x + LINE_WIDTH//2 + 1)
        image[:, start:end] = 0
    
    # 应用高斯模糊
    return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

def forward_transform(image):
    """
    应用多项式畸变并添加噪声
    参数：
        image: 原始输入图像（uint8灰度图）
    返回：
        tuple: (畸变图像, 归一化X坐标矩阵, 归一化Y坐标矩阵, 归一化畸变X坐标矩阵, 归一化畸变Y坐标矩阵)
    """
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    
    # 中心化坐标（归一化到[-1,1]范围）
    x_uni = (x.astype(np.float32) - w/2) / (w/2)
    y_uni = (y.astype(np.float32) - h/2) / (h/2)
    
    # 计算多项式变换
    u, v = compute_polynomial(
        x_uni.ravel(), 
        y_uni.ravel(),
        I=I_MATRIX,
        G=G_MATRIX,
        coeffs_a=POLY_COEFFS_A,
        coeffs_b=POLY_COEFFS_B
    )
    
    # 重塑为与输入坐标相同的形状
    u = u.reshape(h, w)
    v = v.reshape(h, w)
    
    # 添加高斯随机噪声，同样按归一化计算
    u += np.random.normal(0, NOISE_STD/w, (h, w))
    v += np.random.normal(0, NOISE_STD/h, (h, w))
    
    # 换算为像素坐标系
    map_x = (u * (w/2) + w/2).reshape(h, w).astype(np.float32)
    map_y = (v * (h/2) + h/2).reshape(h, w).astype(np.float32)
    
    # 使用OpenCV重映射实现畸变
    distorted_image = cv2.remap(
        image, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=255
    )
    
    return distorted_image, x_uni, y_uni, u, v

def sample_distortion_data(x_uni, y_uni, u, v):
    """
    从归一化坐标数据中随机采样特征点
    参数：
        x_uni: 归一化 X 坐标数组（h,w），范围 [-1, 1]
        y_uni: 归一化 Y 坐标数组（h,w），范围 [-1, 1]
        u: 畸变后 X 坐标数组（h,w），范围 [-1, 1]
        v: 畸变后 Y 坐标数组（h,w），范围 [-1, 1]
    返回：
        采样点坐标元组 (x_src, y_src, u_dst, v_dst)
    """
    total_points = x_uni.size
    sample_num = min(SAMPLE_POINTS, total_points)
    
    # 随机采样索引
    indices = np.random.choice(total_points, sample_num, replace=False)
    
    # 提取采样点（保持二维结构）
    x_samples = x_uni.reshape(-1)[indices]
    y_samples = y_uni.reshape(-1)[indices]
    u_samples = u.reshape(-1)[indices]
    v_samples = v.reshape(-1)[indices]
    
    return x_samples, y_samples, u_samples, v_samples

def estimate_coefficients(x_src, y_src, u_dst, v_dst):
    """从采样点估计畸变系数"""
    from matrix_polynomial_math import solve_poly_coeff
    
    a_coeffs, b_coeffs = solve_poly_coeff(
        x_src, y_src, u_dst, v_dst,
        I=I_MATRIX,
        G=G_MATRIX,
        num_terms=len(POLY_COEFFS_A)
    )
    
    # 打印系数对比
    print("\n真实系数 vs 计算系数：")
    print(f"{'阶数':<5} | {'a_real':<10} {'a_calc':<10} | {'b_real':<10} {'b_calc':<10}")
    for i, (a_real, a_calc, b_real, b_calc) in enumerate(zip(
        POLY_COEFFS_A, a_coeffs, 
        POLY_COEFFS_B, b_coeffs
    )):
        print(f"{i+1:^5} | {a_real:<10.2e} {a_calc:<10.2e} | {b_real:<10.2e} {b_calc:<10.2e}")
    
    return a_coeffs, b_coeffs

def display(original_image, distorted_image, corrected_image):
    """
    可视化显示结果
    参数：
        original_image: 原始图像
        distorted_image: 畸变图像
        corrected_image: 校正结果
    """
    # 配置显示环境
    plt.rc('font', family=FONT_FAMILY, size=FONT_SIZE)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2子图布局
    fig, axs = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]))
    ax1, ax2, ax3, ax4 = axs.ravel()
    
    # 显示原始网格
    ax1.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('原始网格', color=FONT_COLOR)
    ax1.axis('off')
    
    # 显示畸变图像
    ax2.imshow(distorted_image, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('畸变图像', color=FONT_COLOR)
    ax2.axis('off')
    
    # 显示校正结果
    ax3.imshow(corrected_image, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('校正结果', color=FONT_COLOR)
    ax3.axis('off')
    
    # 创建红绿叠加对比图
    org = original_image.astype(float)/255  
    cor = corrected_image.astype(float)/255 
    dis = distorted_image.astype(float)/255 
    overlay = np.dstack((org, cor, dis*0))
    
    ax4.imshow(overlay)
    ax4.set_title('原始(红) vs 校正(绿)', color=FONT_COLOR)
    ax4.axis('off')
    
    # 调整布局参数
    plt.subplots_adjust(
        left=0.05, 
        right=0.95, 
        wspace=0.1, 
        hspace=0.2,
        bottom=0.05, 
        top=0.95
    )
    plt.show()

def generate_grid(cols: int = GRID_COLS, 
                 rows: int = GRID_ROWS, 
                 cell_size: int = CELL_SIZE) -> np.ndarray:
    """
    Generate a grid image for distortion testing.
    
    Args:
        cols: Number of grid columns
        rows: Number of grid rows
        cell_size: Size of each grid cell in pixels
        
    Returns:
        np.ndarray: Grid image (uint8 grayscale)
        
    生成用于畸变测试的网格图像。
    
    参数：
        cols: 网格列数
        rows: 网格行数
        cell_size: 每个网格单元的像素大小
        
    返回：
        np.ndarray: 网格图像（uint8灰度图）
    """
    # 初始化白色背景图像
    image = np.ones((rows*cell_size, cols*cell_size), dtype=np.uint8)*255
    
    # 绘制水平网格线
    for i in range(rows + 1):
        y = i * cell_size
        start = max(0, y - LINE_WIDTH//2)
        end = min(image.shape[0], y + LINE_WIDTH//2 + 1)
        image[start:end, :] = 0
    
    # 绘制垂直网格线
    for j in range(cols + 1):
        x = j * cell_size
        start = max(0, x - LINE_WIDTH//2)
        end = min(image.shape[1], x + LINE_WIDTH//2 + 1)
        image[:, start:end] = 0
    
    # 应用高斯模糊
    return cv2.GaussianBlur(image, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

if __name__ == '__main__':
    total_start = time.time()
    
    # 生成原始图像
    t0 = time.time()
    original_image = create_grid_image()
    print(f'原始图像生成耗时    : {time.time() - t0:.3f} 秒')
    print(f'原始图像尺寸        : {original_image.shape}')

    # 根据预设多项式系数生成畸变图像
    t1 = time.time()
    distorted_image, x_uni, y_uni, u_uni, v_uni = forward_transform(original_image)
    print(f'畸变生成耗时        : {time.time() - t1:.3f} 秒')
    print(f'畸变图像尺寸        : {distorted_image.shape}')
    print(f'坐标数据尺寸        : x_uni={x_uni.shape}, y_uni={y_uni.shape}, u_uni={u_uni.shape}, v_uni={v_uni.shape}')
    
    # 从畸变图像中随机采样特征点
    t2 = time.time()
    x_src, y_src, u_dst, v_dst = sample_distortion_data(x_uni, y_uni, u_uni, v_uni)
    print(f'采样点提取耗时      : {time.time() - t2:.3f} 秒')
    
    # 根据采样点估计多项式系数
    t3 = time.time()
    a_est, b_est = estimate_coefficients(x_src, y_src, u_dst, v_dst)
    print(f'系数求解耗时        : {time.time() - t3:.3f} 秒')
    
    # 根据多项式系数生成校正图像
    t4 = time.time()
    corrected_image = backward_transform(
        distorted_image, 
        a_est, 
        b_est, 
        I=I_MATRIX, 
        G=G_MATRIX,
        interp_range=INTERP_RANGE
    )
    print(f'图像校正耗时        : {time.time() - t4:.3f} 秒')
    
    # 可视化显示
    t5 = time.time()
    display(original_image, distorted_image, corrected_image)
    print(f'显示耗时            : {time.time() - t5:.3f} 秒')
    
    print(f'\n总执行耗时          : {time.time() - total_start:.3f} 秒')
