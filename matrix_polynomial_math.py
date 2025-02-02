"""
Matrix Polynomial Math Module for Image Distortion Correction

This module provides core mathematical functions for polynomial-based image distortion correction.
Key features:
- Matrix decomposition into I and G components
- Generation of polynomial power terms
- Forward and backward polynomial transformations
- High-resolution interpolation

矩阵多项式数学模块

本模块提供基于多项式的图像畸变校正核心数学函数。
主要特性：
- 矩阵分解为I和G分量
- 生成多项式幂次项
- 正向和反向多项式变换
- 高分辨率插值
"""

import numpy as np
from typing import Sequence
import threading
import sys
import os

def decompose_matrix(w: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose matrix w = x*I + y*G into its (x,y) components.
    For a given matrix w and basis matrix G, find the unique coefficients x and y
    such that w = x*I + y*G, where I is the identity matrix.
    
    Args:
        w (np.ndarray): Array of matrices in form x*I + y*G
            Shape: (2, 2, N) where N is the number of points.
        G (np.ndarray): The basis matrix G of shape (2, 2).
        
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x (np.ndarray): Coefficients of I, shape (N,)
            - y (np.ndarray): Coefficients of G, shape (N,)
            
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If matrix shapes are incorrect
    
    将矩阵 w = x*I + y*G 分解为其 (x,y) 分量。
    对于给定的矩阵 w 和基底矩阵 G，找到唯一的系数 x 和 y，使得 w = x*I + y*G，
    其中 I 为单位矩阵。
    
    参数：
        w (np.ndarray): 形如 x*I + y*G 的矩阵数组
            形状：(2, 2, N)，其中 N 是点的数量
        G (np.ndarray): 形状为 (2, 2) 的基底矩阵 G
        
    返回值：
        tuple[np.ndarray, np.ndarray]：包含以下内容的元组：
            - x (np.ndarray)：I 的系数，形状为 (N,)
            - y (np.ndarray)：G 的系数，形状为 (N,)
            
    异常：
        TypeError：如果输入不是 numpy 数组
        ValueError：如果矩阵形状不正确
    """
    if not isinstance(G, np.ndarray):
        raise TypeError(f"G must be a numpy array, got {type(G)}")
    if G.shape != (2, 2):
        raise ValueError(f"G must have shape (2, 2), got {G.shape}")
    
    if not isinstance(w, np.ndarray):
        raise TypeError(f"w must be a numpy array, got {type(w)}")
    if w.ndim != 3 or w.shape[0:2] != (2, 2):
        raise ValueError(f"w must have shape (2, 2, N), got {w.shape}")
    
    x = np.trace(w)/2  # Extract coefficient of I
    y = np.trace(G.transpose() @ w)/np.trace(G @ G.transpose())  # Extract coefficient of G
    return x, y

def generate_powers(x: np.ndarray, y: np.ndarray, I: np.ndarray, G: np.ndarray, 
                   num_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate base matrices in terms of f and g for the polynomial computation.
    For each power i, compute (xI + yG)^i = f[i,:]*I + g[i,:]*G, where:
    - f[0,:] = x (first order term)
    - g[0,:] = y (first order term)
    Higher powers are computed through non-commutative matrix multiplication.
    
    Args:
        x (np.ndarray): x-coordinates, shape (N,)
        y (np.ndarray): y-coordinates, shape (N,)
        I (np.ndarray): 2x2 identity matrix
        G (np.ndarray): 2x2 basis matrix
        num_terms (int): Number of terms in polynomial (excluding constant term)
        
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - f (np.ndarray): Coefficients of I for each power, shape (num_terms, N)
            - g (np.ndarray): Coefficients of G for each power, shape (num_terms, N)
            
    Raises:
        TypeError: If inputs have wrong types
        ValueError: If shapes are incorrect or num_terms < 1
    
    为多项式计算生成基底矩阵的 f 和 g 分量。
    对于每个幂次 i，计算 (xI + yG)^i = f[i,:]*I + g[i,:]*G，其中：
    - f[0,:] = x（一阶项）
    - g[0,:] = y（一阶项）
    更高次幂通过非交换矩阵乘法计算。
    
    参数：
        x (np.ndarray)：x 坐标，形状为 (N,)
        y (np.ndarray)：y 坐标，形状为 (N,)
        I (np.ndarray)：2x2 单位矩阵
        G (np.ndarray)：2x2 基底矩阵
        num_terms (int)：多项式的项数（不包括常数项）
        
    返回值：
        tuple[np.ndarray, np.ndarray]：
            - f (np.ndarray)：每个幂次的 I 系数，形状为 (num_terms, N)
            - g (np.ndarray)：每个幂次的 G 系数，形状为 (num_terms, N)
            
    异常：
        TypeError：如果输入类型错误
        ValueError：如果形状不正确或 num_terms < 1
    """
    # Check input vectors
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"x and y must be numpy arrays, got {type(x)} and {type(y)}")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"x and y must be 1-dimensional, got {x.ndim} and {y.ndim} dimensions")
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    # Check matrices
    if not isinstance(I, np.ndarray) or not isinstance(G, np.ndarray):
        raise TypeError(f"I and G must be numpy arrays, got {type(I)} and {type(G)}")
    if I.shape != (2, 2) or G.shape != (2, 2):
        raise ValueError(f"I and G must have shape (2, 2), got {I.shape} and {G.shape}")
    
    if not isinstance(num_terms, int):
        raise TypeError(f"num_terms must be an integer, got {type(num_terms)}")
    if num_terms < 1:
        raise ValueError(f"num_terms must be at least 1, got {num_terms}")
    
    N = len(x)
    f = np.zeros((num_terms, N))  # Coefficients of I
    g = np.zeros((num_terms, N))  # Coefficients of G
    
    # First power (i=1)
    w = np.outer(np.ravel(I), x) + np.outer(np.ravel(G), y)  # w = xI + yG
    w = np.reshape(w, (2, 2, N))
    f[0], g[0] = decompose_matrix(w, G)
    
    # Higher powers through matrix multiplication
    if num_terms > 1:
        w1 = w.copy()
        for i in range(1, num_terms):
            w1 = np.einsum('ijk,jlk->ilk', w1, w)  # Matrix multiplication
            f[i], g[i] = decompose_matrix(w1, G)
    
    return f, g

def compute_polynomial(x: np.ndarray, y: np.ndarray, I: np.ndarray, G: np.ndarray,
                      coeffs_a: Sequence[float], coeffs_b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the matrix polynomial P(xI + yG) = (xI + yG) + sum(a_i*I + b_i*G)(xI + yG)^i.
    The first terms (i=1) correspond to the identity transformation (x,y),
    followed by higher order terms with coefficients a_i and b_i.
    
    Args:
        x (np.ndarray): x-coordinates of input points, shape (N,)
        y (np.ndarray): y-coordinates of input points, shape (N,)
        I (np.ndarray): 2x2 identity matrix
        G (np.ndarray): 2x2 basis matrix
        coeffs_a (Sequence[float]): Coefficients for I terms
        coeffs_b (Sequence[float]): Coefficients for G terms
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Transformed (x, y) coordinates
        
    Raises:
        TypeError: If inputs have wrong types
        ValueError: If shapes are incorrect or coefficients lists have different lengths

    计算矩阵多项式 P(xI + yG) = (xI + yG) + sum(a_i*I + b_i*G)(xI + yG)^i。
    第一项（i=1）对应于单位变换 (x,y)，
    后面是带系数 a_i 和 b_i 的高阶项。
    
    参数：
        x (np.ndarray)：输入点的 x 坐标，形状为 (N,)
        y (np.ndarray)：输入点的 y 坐标，形状为 (N,)
        I (np.ndarray)：2x2 单位矩阵
        G (np.ndarray)：2x2 基底矩阵
        coeffs_a (Sequence[float])：I 项的系数
        coeffs_b (Sequence[float])：G 项的系数
    
    返回值：
        tuple[np.ndarray, np.ndarray]：变换后的 (x, y) 坐标
        
    异常：
        TypeError：如果输入类型错误或系数不是 lists 或 numpy arrays
        ValueError：如果形状不正确或系数列表长度不同
    """
    # Check input vectors
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"x and y must be numpy arrays, got {type(x)} and {type(y)}")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"x and y must be 1-dimensional, got {x.ndim} and {y.ndim} dimensions")
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    # Check matrices
    if not isinstance(I, np.ndarray) or not isinstance(G, np.ndarray):
        raise TypeError(f"I and G must be numpy arrays, got {type(I)} and {type(G)}")
    if I.shape != (2, 2) or G.shape != (2, 2):
        raise ValueError(f"I and G must have shape (2, 2), got {I.shape} and {G.shape}")
    
    # Check coefficients
    if not isinstance(coeffs_a, (list, np.ndarray)) or not isinstance(coeffs_b, (list, np.ndarray)):
        raise TypeError(f"coefficients must be lists or numpy arrays")
    if len(coeffs_a) == 0 or len(coeffs_b) == 0:
        raise ValueError("coefficients cannot be empty")
    if len(coeffs_a) != len(coeffs_b):
        raise ValueError(f"coeffs_a and coeffs_b must have same length, got {len(coeffs_a)} and {len(coeffs_b)}")
    
    # Generate base matrices for each power (starting from power 1)
    f, g = generate_powers(x, y, I, G, len(coeffs_a))
    
    # Convert coefficients to arrays for matrix multiplication
    coeffs_a = np.array(coeffs_a)
    coeffs_b = np.array(coeffs_b)
    
    # Compute result directly using matrix multiplication
    u = coeffs_a @ f + coeffs_b @ g  # Coefficient of I
    v = coeffs_b @ f + coeffs_a @ g  # Coefficient of G
    
    return u, v 

def solve_poly_coeff(x_src: np.ndarray, y_src: np.ndarray, u_dst: np.ndarray, v_dst: np.ndarray,
                    I: np.ndarray, G: np.ndarray, num_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve polynomial coefficients from corresponding points.
    For given source points (x_src, y_src) and their distorted positions (u_dst, v_dst),
    find the polynomial coefficients that best describe the transformation.
    
    Args:
        x_src (np.ndarray): Source x-coordinates, normalized to [-1,1], shape (N,)
        y_src (np.ndarray): Source y-coordinates, normalized to [-1,1], shape (N,)
        u_dst (np.ndarray): Distorted x-coordinates, normalized to [-1,1], shape (N,)
        v_dst (np.ndarray): Distorted y-coordinates, normalized to [-1,1], shape (N,)
        I (np.ndarray): 2x2 identity matrix
        G (np.ndarray): 2x2 basis matrix
        num_terms (int): Number of polynomial terms
    
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - Coefficients for I terms (a_coeffs)
            - Coefficients for G terms (b_coeffs)
            
    Raises:
        TypeError: If inputs have wrong types
        ValueError: If shapes are incorrect
    
    从对应点求解多项式系数。
    对于给定的源点 (x_src, y_src) 和其畸变位置 (u_dst, v_dst)，
    找到最佳描述该变换的多项式系数。
    
    参数：
        x_src (np.ndarray)：源 x 坐标，归一化到 [-1,1]，形状为 (N,)
        y_src (np.ndarray)：源 y 坐标，归一化到 [-1,1]，形状为 (N,)
        u_dst (np.ndarray)：畸变 x 坐标，归一化到 [-1,1]，形状为 (N,)
        v_dst (np.ndarray)：畸变 y 坐标，归一化到 [-1,1]，形状为 (N,)
        I (np.ndarray)：2x2 单位矩阵
        G (np.ndarray)：2x2 基底矩阵
        num_terms (int)：多项式项数
    
    返回值：
        tuple[np.ndarray, np.ndarray]：
            - I 项的系数 (a_coeffs)
            - G 项的系数 (b_coeffs)
            
    异常：
        TypeError：如果输入类型错误
        ValueError：如果形状不正确
    """
    # Check input vectors
    if not isinstance(x_src, np.ndarray) or not isinstance(y_src, np.ndarray):
        raise TypeError(f"x_src and y_src must be numpy arrays, got {type(x_src)} and {type(y_src)}")
    if x_src.ndim != 1 or y_src.ndim != 1:
        raise ValueError(f"x_src and y_src must be 1-dimensional, got {x_src.ndim} and {y_src.ndim} dimensions")
    if len(x_src) != len(y_src):
        raise ValueError(f"x_src and y_src must have same length, got {len(x_src)} and {len(y_src)}")
    
    # Generate polynomial terms matrix
    f, g = generate_powers(
        x_src,  
        y_src,
        I=I,
        G=G,
        num_terms=num_terms
    )

    # Build linear system
    f_p_g = (f + g).transpose()  # f+g terms matrix transposed
    f_m_g = (f - g).transpose()  # f-g terms matrix transposed
    u_p_v = (u_dst + v_dst).ravel()  # u+v vector
    u_m_v = (u_dst - v_dst).ravel()  # u-v vector

    # Solve least squares
    a_p_b, _, _, _ = np.linalg.lstsq(f_p_g, u_p_v, rcond=None)
    a_m_b, _, _, _ = np.linalg.lstsq(f_m_g, u_m_v, rcond=None)

    # Decouple coefficients
    return (a_p_b + a_m_b)/2, (a_p_b - a_m_b)/2

def backward_transform(img: np.ndarray, a_est: np.ndarray, b_est: np.ndarray,
                      I: np.ndarray, G: np.ndarray, interp_range: int = 2) -> np.ndarray:
    """
    Apply inverse polynomial transformation to correct image distortion.
    Uses high-resolution interpolation to improve accuracy.
    
    Args:
        img (np.ndarray): Input distorted image (uint8 grayscale)
        a_est (np.ndarray): Estimated coefficients for I terms
        b_est (np.ndarray): Estimated coefficients for G terms
        I (np.ndarray): 2x2 identity matrix
        G (np.ndarray): 2x2 basis matrix
        interp_range (int, optional): Interpolation neighborhood range. Defaults to 2
    
    Returns:
        np.ndarray: Corrected image (uint8 format)
        
    Raises:
        TypeError: If inputs have wrong types
        ValueError: If shapes are incorrect
    
    应用逆多项式变换校正图像畸变。
    使用高分辨率插值以提高精度。
    
    参数：
        img (np.ndarray)：输入的畸变图像（uint8 灰度图）
        a_est (np.ndarray)：I 项的估计系数
        b_est (np.ndarray)：G 项的估计系数
        I (np.ndarray)：2x2 单位矩阵
        G (np.ndarray)：2x2 基底矩阵
        interp_range (int, optional)：插值邻域范围，默认为 2
    
    返回值：
        np.ndarray：校正后的图像（uint8 格式）
        
    异常：
        TypeError：如果输入类型错误
        ValueError：如果形状不正确
    """
    # Check inputs
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a numpy array, got {type(img)}")
    if img.ndim != 2:
        raise ValueError(f"img must be 2-dimensional, got {img.ndim} dimensions")
    
    scale = interp_range
    h0, w0 = img.shape

    # Generate high resolution image
    img_high = np.kron(img, np.ones((scale, scale))).astype(np.float32)
    weight_high = np.ones_like(img_high)
    h, w = img_high.shape
    
    # Generate coordinate grid
    y, x = np.mgrid[:h, :w]
    
    # Normalize coordinates to [-1,1] range
    x_uni = (x.astype(np.float32) - w/2) / (w/2)
    y_uni = (y.astype(np.float32) - h/2) / (h/2)
    
    # Compute inverse transformation coordinates
    u, v = compute_polynomial( 
        x_uni.ravel(), y_uni.ravel(),
        I=I, G=G,
        coeffs_a=a_est, coeffs_b=b_est
    )
    
    # Convert back to pixel coordinates
    u_float = (u * w/2 + w/2).reshape(h, w)
    v_float = (v * h/2 + h/2).reshape(h, w)
    
    # Inline interpolation function
    def run_interpolation(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Perform bilinear interpolation.
        
        Args:
            img (np.ndarray): Source image
            u (np.ndarray): x coordinates
            v (np.ndarray): y coordinates
            
        Returns:
            np.ndarray: Interpolated values
        """
        results = np.zeros((4, img.shape[0], img.shape[1]), dtype=np.float32)
        u_floor = np.floor(u).astype(int).clip(0, img.shape[1]-2)
        v_floor = np.floor(v).astype(int).clip(0, img.shape[0]-2)
        
        # Calculate relative offsets
        dx = np.abs(u - u_floor)
        dy = np.abs(v - v_floor)

        # Define arrays for parallel computation
        shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]
        weights = np.zeros((4, img.shape[0], img.shape[1]), dtype=np.float32)
        weights[0] = (1-dy)*(1-dx)*img
        weights[1] = (1-dy)*dx*img
        weights[2] = dy*(1-dx)*img
        weights[3] = dy*dx*img

        # Parallel computation of corner weights
        def start_thread(i: int):
            y_shift, x_shift = shifts[i]    
            np.add.at(results[i], (v_floor+y_shift, u_floor+x_shift), weights[i])
        
        threads = [
            threading.Thread(target=start_thread, args=(i,))
            for i in range(4)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        return sum(results)
    
    # Perform bilinear interpolation
    fixed_high = run_interpolation(img_high, u_float, v_float)
    weight_high = run_interpolation(weight_high, u_float, v_float)
    
    # Downsample
    fixed_low = fixed_high.reshape(h0, scale, w0, scale).sum(axis=(1,3))
    weight_low = weight_high.reshape(h0, scale, w0, scale).sum(axis=(1,3))
    
    # Weighted average for final result
    with np.errstate(divide='ignore', invalid='ignore'):
        fixed_low = np.divide(fixed_low, weight_low, where=weight_low!=0)
        fixed_low[weight_low == 0] = 0
    
    result = np.clip(fixed_low, 0, 255).astype(np.uint8)
    
    return result 