# -*- coding: utf-8 -*-
"""
网格畸变校正系统
功能：
1. 生成标准网格图像
2. 应用多项式畸变并添加噪声
3. 采样特征点估计畸变参数
4. 执行图像校正
5. 可视化对比结果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage
import cv2
from matrix_polynomial_math import generate_powers
import time
import threading  # 确保该导入语句存在

# ===================== 全局可调参数 =====================
INTERP_RANGE = 2        # 插值邻域范围，出现细碎空区时增大该值

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
GRID_ROWS = 6            # 网格纵向单元格数量（Y方向）
GRID_COLS = 8            # 网格横向单元格数量（X方向）
CELL_SIZE = 200          # 单元格边长（像素）
LINE_WIDTH = 1           # 网格线宽度（像素）

# 显示参数
FIGURE_SIZE = (12, 8)    # 显示窗口物理尺寸（宽, 高，单位：英寸）
DISPLAY_DPI = 300         # 图像显示分辨率（每英寸点数）
FONT_FAMILY = 'Microsoft YaHei'  # 字体名称（需系统支持）
FONT_SIZE = 8            # 字体大小（磅）
FONT_COLOR = 'black'     # 标题字体颜色

# ===================== 核心功能实现 =====================
def create_grid():
    """
    生成标准网格图像
    返回：
        uint8格式的网格图像矩阵（白底黑线）
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
    
    return image

def apply_polynomial_distortion(image):
    """
    应用多项式畸变并添加噪声
    参数：
        image: 原始输入图像（uint8灰度图）
    返回：
        tuple: (畸变图像, 归一化X坐标矩阵, 归一化Y坐标矩阵, 畸变X坐标矩阵, 畸变Y坐标矩阵)
    """
    from matrix_polynomial_math import compute_polynomial
    
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    
    # 中心化坐标（归一化到[-1,1]范围）
    x_sym = (x.astype(np.float32) - w/2) / (w/2)
    y_sym = (y.astype(np.float32) - h/2) / (h/2)
    
    # 计算多项式变换
    u, v = compute_polynomial(
        x_sym.ravel(), 
        y_sym.ravel(),
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
    
    # 转换回像素坐标系
    map_x = (u * (w/2) + w/2).reshape(h, w).astype(np.float32)
    map_y = (v * (h/2) + h/2).reshape(h, w).astype(np.float32)
    
    # 使用OpenCV重映射实现畸变
    distorted = cv2.remap(
        image, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=255
    )
    
    return distorted, x_sym, y_sym, u, v

def solve_distortion_coefficients(x_src, y_src, u_dst, v_dst):
    """
    从对应点求解畸变多项式系数
    参数：
        x_src: 原始X坐标（归一化到[-1,1]的1D数组）
        y_src: 原始Y坐标（归一化到[-1,1]的1D数组）
        u_dst: 畸变X坐标（归一化到[-1,1]的1D数组）
        v_dst: 畸变Y坐标（归一化到[-1,1]的1D数组）
    返回：
        tuple: (A系数数组, B系数数组)
    """
    # 添加形状验证
    assert x_src.ndim == 1, "输入坐标应为1D数组"
    assert y_src.shape == x_src.shape, "坐标维度不匹配"
    
    num_terms = len(POLY_COEFFS_A) 
    
    # 生成多项式项矩阵
    f, g = generate_powers(
        x_src,  
        y_src,
        I=I_MATRIX,
        G=G_MATRIX,
        num_terms=num_terms
    )

    # 构建线性方程组
    f_p_g = (f + g).transpose()  # f+g项矩阵转置
    f_m_g = (f - g).transpose()  # f-g项矩阵转置
    u_p_v = (u_dst + v_dst).ravel()  # u+v向量
    u_m_v = (u_dst - v_dst).ravel()  # u-v向量

    # 最小二乘求解
    a_p_b, _, _, _ = np.linalg.lstsq(f_p_g, u_p_v, rcond=None)
    a_m_b, _, _, _ = np.linalg.lstsq(f_m_g, u_m_v, rcond=None)

    # 解耦系数
    return (a_p_b + a_m_b)/2, (a_p_b - a_m_b)/2

def generate_original_grid():
    """
    生成带高斯模糊的原始网格
    返回：
        高斯模糊处理后的网格图像（uint8格式）
    """
    grid = create_grid()
    return cv2.GaussianBlur(grid, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

def sample_distortion_data(x_sym, y_sym, u, v):
    """
    从归一化坐标数据中随机采样特征点
    参数：
        x_sym: 归一化 X 坐标数组（h,w），范围 [-1, 1]
        y_sym: 归一化 Y 坐标数组（h,w），范围 [-1, 1]
        u: 畸变后 X 坐标数组（h,w），范围 [-1, 1]
        v: 畸变后 Y 坐标数组（h,w），范围 [-1, 1]
    返回：
        采样点坐标元组 (x_src, y_src, u_dst, v_dst)
    """
    total_points = x_sym.size
    sample_num = min(SAMPLE_POINTS, total_points)
    
    # 随机采样索引
    indices = np.random.choice(total_points, sample_num, replace=False)
    
    # 提取采样点（保持二维结构）
    x_samples = x_sym.reshape(-1)[indices]
    y_samples = y_sym.reshape(-1)[indices]
    u_samples = u.reshape(-1)[indices]
    v_samples = v.reshape(-1)[indices]
    
    return x_samples, y_samples, u_samples, v_samples

def estimate_coefficients(x_src, y_src, u_dst, v_dst):
    """从采样点估计畸变系数"""
    a_coeffs, b_coeffs = solve_distortion_coefficients(x_src, y_src, u_dst, v_dst)
    
    # 打印系数对比
    print("\n真实系数 vs 计算系数：")
    print(f"{'阶数':<5} | {'a_real':<10} {'a_calc':<10} | {'b_real':<10} {'b_calc':<10}")
    for i, (a_real, a_calc, b_real, b_calc) in enumerate(zip(
        POLY_COEFFS_A, a_coeffs, 
        POLY_COEFFS_B, b_coeffs
    )):
        print(f"{i+1:^5} | {a_real:<10.2e} {a_calc:<10.2e} | {b_real:<10.2e} {b_calc:<10.2e}")
    
    return a_coeffs, b_coeffs

def run_interpolation(img, u, v):
    """
    双线性插值实现（实验性多线程版本）
    注意：由于Python的GIL限制，实际加速效果可能有限
    参数：
        img: 输入图像矩阵（H x W）
        u: 目标点x坐标矩阵（浮点型，H x W）
        v: 目标点y坐标矩阵（浮点型，H x W）
    返回：
        插值后的图像矩阵（H x W）
    """
    results = [np.zeros_like(img, dtype=np.float32) for _ in range(4)]
    u_floor = np.floor(u).astype(int).clip(0, img.shape[1]-2)
    v_floor = np.floor(v).astype(int).clip(0, img.shape[0]-2)
    
    # 计算相对偏移量
    dx = np.abs(u - u_floor)
    dy = np.abs(v - v_floor)
    
    def add_contribution(result, coords, weights):
        """线程工作函数"""
        np.add.at(result, coords, weights * img)
    
    # 创建四个线程
    threads = []
    threads.append(threading.Thread(target=add_contribution, 
                                 args=(results[0], (v_floor, u_floor), (1-dy) * (1-dx))))
    threads.append(threading.Thread(target=add_contribution, 
                                 args=(results[1], (v_floor, u_floor+1), (1-dy) * dx)))
    threads.append(threading.Thread(target=add_contribution, 
                                 args=(results[2], (v_floor+1, u_floor), dy * (1-dx))))
    threads.append(threading.Thread(target=add_contribution, 
                                 args=(results[3], (v_floor+1, u_floor+1), dy * dx)))
    
    # 启动所有线程
    for thread in threads:
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 合并结果
    result = sum(results)
    
    return result

def apply_correction(original, a_est, b_est):
    """
    应用逆多项式变换进行图像校正
    实现策略：
        1. 升采样提高插值精度
        2. 计算逆变换坐标
        3. 双线性插值获取高分辨率结果
        4. 降采样恢复原始分辨率
    参数：
        original: 待校正图像（uint8灰度图）
        a_est: 估计的逆变换多项式A系数
        b_est: 估计的逆变换多项式B系数
    返回：
        校正后的图像（uint8格式）
    """
    from matrix_polynomial_math import compute_polynomial
    
    scale = INTERP_RANGE
    h0, w0 = original.shape

    # 生成高分辨率图像
    img_high = np.kron(original, np.ones((scale, scale))).astype(np.float32)
    weight_high = np.ones_like(img_high)
    h, w = img_high.shape
    
    # 生成坐标网格
    y, x = np.mgrid[:h, :w]
    
    # 归一化坐标到[-1,1]范围
    x_sym = (x.astype(np.float32) - w/2) / (w/2)
    y_sym = (y.astype(np.float32) - h/2) / (h/2)
    
    # 计算逆变换坐标
    u, v = compute_polynomial( 
        x_sym.ravel(), y_sym.ravel(),
        I=I_MATRIX, G=G_MATRIX,
        coeffs_a=a_est, coeffs_b=b_est
    )
    
    # 转换回像素坐标系
    u_float = (u * (w/2) + w/2).reshape(h, w)
    v_float = (v * (h/2) + h/2).reshape(h, w)
    
    # 执行双线性插值
    fixed_high = run_interpolation(img_high, u_float, v_float)
    weight_high = run_interpolation(weight_high, u_float, v_float)
    
    # 降采样处理
    fixed_low = fixed_high.reshape(h0, scale, w0, scale).sum(axis=(1,3))
    weight_low = weight_high.reshape(h0, scale, w0, scale).sum(axis=(1,3))
    
    # 加权平均计算最终结果
    with np.errstate(divide='ignore', invalid='ignore'):
        fixed_low = np.divide(fixed_low, weight_low, where=weight_low!=0)
        fixed_low[weight_low == 0] = 0
    
    # if np.any(weight_low == 0):
    #     print("警告                : 检测到权重为 0 的像素，可能导致图像空洞")

    result = np.clip(fixed_low, 0, 255).astype(np.uint8)
    
    return result

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

if __name__ == '__main__':
    total_start = time.time()
    
    # 数据生成流水线
    t0 = time.time()
    original_image = generate_original_grid()
    print(f'原始图像生成时间    : {time.time() - t0:.3f} 秒')
    print(f'原始图像尺寸        : {original_image.shape}')

    # 畸变数据生成
    t1 = time.time()
    distorted_image, x_sym, y_sym, u, v = apply_polynomial_distortion(original_image)
    print(f'畸变生成时间        : {time.time() - t1:.3f} 秒')
    print(f'畸变图像尺寸        : {distorted_image.shape}')
    print(f'坐标数据尺寸        : x_sym={x_sym.shape}, y_sym={y_sym.shape}, u={u.shape}, v={v.shape}')
    
    # 多项式系数估计流程
    t2 = time.time()
    x_src, y_src, u_dst, v_dst = sample_distortion_data(x_sym, y_sym, u, v)
    print(f'采样点提取时间      : {time.time() - t2:.3f} 秒')
    
    t3 = time.time()
    a_est, b_est = estimate_coefficients(x_src, y_src, u_dst, v_dst)
    print(f'系数求解时间        : {time.time() - t3:.3f} 秒')
    
    # 图像校正流程
    t4 = time.time()
    corrected_image = apply_correction(distorted_image, a_est, b_est)
    print(f'图像校正时间        : {time.time() - t4:.3f} 秒')
    
    # 可视化显示
    t5 = time.time()
    display(original_image, distorted_image, corrected_image)
    print(f'显示时间            : {time.time() - t5:.3f} 秒')
    
    print(f'\n总执行时间          : {time.time() - total_start:.3f} 秒')
