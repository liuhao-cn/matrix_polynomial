"""
Matrix Polynomial Math Module

This module provides core mathematical functions for matrix polynomial calculations.
The main functionality includes:
- Matrix decomposition into I and G components
- Generation of polynomial power terms
- Computation of matrix polynomials
- Grid point generation for visualization
+ Basic matrix polynomial computations:
+   - Matrix decomposition into I and G components
+   - Generation of polynomial power terms
+   - Computation of matrix polynomials

矩阵多项式数学模块

本模块提供矩阵多项式计算的核心数学函数。
主要功能包括：
- 将矩阵分解为 I 和 G 分量
- 生成多项式幂次项
- 计算矩阵多项式
- 生成用于可视化的网格点
+ 基础矩阵多项式计算：
+   - 将矩阵分解为 I 和 G 分量
+   - 生成多项式幂次项
+   - 计算矩阵多项式
"""

import numpy as np
from typing import Sequence

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