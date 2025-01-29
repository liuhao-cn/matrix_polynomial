"""
Matrix Polynomial Visualization Tool (Optimized Version)

矩阵多项式可视化工具（优化版）

This module provides a visualization tool for matrix polynomials of the form:
P(x*I + y*G) = sum(a_i*I + b_i*G)(x*I + y*G)^i, where I is the identity matrix 
and G is a user-defined 2x2 matrix. The tool shows how these polynomials transform 
different grid patterns in the complex plane.

本模块提供了一个矩阵多项式可视化工具，用于可视化形如：
P(x*I + y*G) = sum(a_i*I + b_i*G)(x*I + y*G)^i 的矩阵多项式，其中I为单位矩阵，
G为用户定义的2x2矩阵。该工具展示这些多项式如何变换复平面中的不同网格模式。

Key Features:
    - Interactive GUI for matrix G and polynomial coefficients input
    - Six grid pattern options: H+V, H, V, C+R, C, R
    - Real-time visualization with input-output plane comparison
    - Optional 45° rotation for better visualization
    - Auto-scaling plot ranges for transformed patterns

主要特点：
    - 用于输入矩阵G和多项式系数的交互式界面
    - 六种网格模式：水平+垂直、水平、垂直、圆形+径向、圆形、径向
    - 实时可视化，支持输入-输出平面对比
    - 可选的45度旋转以获得更好的可视化效果
    - 自动缩放变换后图案的显示范围
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import os
import platform

# Grid Generation Constants
GRID_RANGE = (-2, 2)  # Range for x and y coordinates
NUM_POINTS = 10000  # Number of points for line generation
MIN_CIRCLE_RADIUS = 0.1  # Minimum radius for circular grids
MAX_CIRCLE_RADIUS = 2.0  # Maximum radius for circular grids
ALPHA_FEW_LINES = 1.0
ALPHA_MANY_LINES = 0.6

# GUI Constants
# Windows-specific GUI parameters
WIN_FONT_SIZE = 23
WIN_BUTTON_PADDING = (10, 0)
WIN_ENTRY_WIDTH = 7  # Increased from 6 to 7 (about 20% more)
WIN_PLOT_DPI = 600
WIN_PLOT_LINEWIDTH = 0.7
WIN_PLOT_FONTSIZE = 5
WIN_FIGURE_SIZE = (4, 2)

# Linux-specific GUI parameters
LINUX_FONT_SIZE = 11
LINUX_BUTTON_PADDING = (5, 0)
LINUX_ENTRY_WIDTH = 5  # Increased from 4 to 5 (about 25% more)
LINUX_PLOT_DPI = 150
LINUX_PLOT_LINEWIDTH = 1.0
LINUX_PLOT_FONTSIZE = 8
LINUX_FIGURE_SIZE = (8, 4)

# Common parameters
FONT_FAMILY = 'Arial'
MATRIX_SIZE = 2
NUM_COEFFICIENTS = 6
COEFFICIENTS_PER_FRAME = 2
PLOT_MARGIN = 1.1  # Plot range margin (10% extra)

# Grid Type Constants
GRID_TYPE_HV = 'Horizontal+Vertical'
GRID_TYPE_H = 'Horizontal'
GRID_TYPE_V = 'Vertical'
GRID_TYPE_NONE = 'None'
GRID_TYPE_RC = 'Circular+Radial'
GRID_TYPE_R = 'Radial'
GRID_TYPE_C = 'Circular'

# Line Style Constants
LINE_STYLE_SOLID = '-'
LINE_STYLE_DOTTED = '-'  # Changed from ':' to '--' for dashed style


def decompose_HyperComplexNumbers(w: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose hypercomplex numbers w = x*I + y*G into their (x,y) components.
    
    Args:
        w (np.ndarray): Array of hypercomplex numbers in matrix form.
            Shape: (2, 2, N) where N is the number of points.
        G (np.ndarray): The basis matrix G of shape (2, 2).
        
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x (np.ndarray): Coefficients of I, shape (N,)
            - y (np.ndarray): Coefficients of G, shape (N,)

    将超复数 w = x*I + y*G 分解为其(x,y)分量。
    
    参数:
        w (np.ndarray): 矩阵形式的超复数数组
            形状: (2, 2, N)，其中N是点的数量
        G (np.ndarray): 基底矩阵G，形状为(2, 2)
        
    返回值:
        tuple[np.ndarray, np.ndarray]: 包含以下内容的元组:
            - x (np.ndarray): I的系数，形状为(N,)
            - y (np.ndarray): G的系数，形状为(N,)
    """
    x = np.trace(w)/2  # Extract coefficient of I / 提取I的系数
    y = np.trace(G.transpose() @ w)/2  # Extract coefficient of G / 提取G的系数
    return x, y


def compute_polynomial(x: np.ndarray, y: np.ndarray, I: np.ndarray, G: np.ndarray, 
                      coeffs_a: list[float], coeffs_b: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the matrix polynomial P(x*I + y*G) = sum(a_i*I + b_i*G)(x*I + y*G)^i at given points.
    
    Args:
        x (np.ndarray): x-coordinates of input points, shape (N,)
        y (np.ndarray): y-coordinates of input points, shape (N,)
        I (np.ndarray): 2x2 identity matrix, shape (2, 2)
        G (np.ndarray): User-defined 2x2 basis matrix, shape (2, 2)
        coeffs_a (list[float]): Coefficients for I terms, length NUM_COEFFICIENTS
        coeffs_b (list[float]): Coefficients for G terms, length NUM_COEFFICIENTS
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - u (np.ndarray): x-coordinates after transformation, shape (N,)
            - v (np.ndarray): y-coordinates after transformation, shape (N,)

    在给定点计算矩阵多项式 P(x*I + y*G) = sum(a_i*I + b_i*G)(x*I + y*G)^i 的值。
    
    参数:
        x (np.ndarray): 输入点的x坐标，形状为(N,)
        y (np.ndarray): 输入点的y坐标，形状为(N,)
        I (np.ndarray): 2x2单位矩阵，形状为(2, 2)
        G (np.ndarray): 用户定义的2x2基底矩阵，形状为(2, 2)
        coeffs_a (list[float]): I项的系数，长度为NUM_COEFFICIENTS
        coeffs_b (list[float]): G项的系数，长度为NUM_COEFFICIENTS
    
    返回值:
        tuple[np.ndarray, np.ndarray]: 包含以下内容的元组:
            - u (np.ndarray): 变换后的x坐标，形状为(N,)
            - v (np.ndarray): 变换后的y坐标，形状为(N,)
    """
    result = np.zeros((MATRIX_SIZE, MATRIX_SIZE, NUM_POINTS))  # Initialize result array / 初始化结果数组
    
    for i in range(NUM_COEFFICIENTS):
        if i==0:
            w = np.outer(np.ravel(I), x) + np.outer(np.ravel(G), y)  # w = x*I + y*G
            w = np.reshape(w, (2, 2, NUM_POINTS))
            w1 = w.copy()
        else:
            w1 = np.einsum('ijk,jlk->ilk', w1, w)  # Matrix multiplication / 矩阵乘法
        
        coeff_matrix = coeffs_a[i] * I + coeffs_b[i] * G  # Current coefficient matrix / 当前系数矩阵
        result += coeff_matrix @ w1  # Add term to sum / 将项加入求和
    
    u, v = decompose_HyperComplexNumbers(result, G)  # Extract coordinates / 提取坐标
    return u, v


class MatrixPolynomialAppOptimized:
    """
    Optimized version of the Matrix Polynomial visualization tool.
    
    This class provides a graphical interface for visualizing matrix polynomial 
    transformations with various grid patterns and interactive controls. It uses
    multiprocessing for faster computation of transformations.

    矩阵多项式可视化工具的优化版本。
    
    该类提供了图形界面，用于可视化具有各种网格模式的矩阵多项式变换，并提供交互式控制。
    使用多进程实现更快的变换计算。
    """
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the Matrix Polynomial visualization tool.
        
        Args:
            root (tk.Tk): The main window of the application

        初始化矩阵多项式可视化工具。
        
        参数:
            root (tk.Tk): 应用程序的主窗口
        """
        self.root = root
        self.root.title("Matrix Polynomial (Optimized)")
        
        # Set config file path
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix_polynomial_config.json')
        
        # Load saved configuration or use defaults
        self.config = self.load_config()
        
        # Determine platform-specific parameters
        is_linux = platform.system() == 'Linux'
        # Use appropriate parameters based on platform
        self.current_font_size = LINUX_FONT_SIZE if is_linux else WIN_FONT_SIZE
        self.current_padding = LINUX_BUTTON_PADDING if is_linux else WIN_BUTTON_PADDING
        self.current_entry_width = LINUX_ENTRY_WIDTH if is_linux else WIN_ENTRY_WIDTH
        self.current_plot_dpi = LINUX_PLOT_DPI if is_linux else WIN_PLOT_DPI
        self.current_plot_linewidth = LINUX_PLOT_LINEWIDTH if is_linux else WIN_PLOT_LINEWIDTH
        self.current_plot_fontsize = LINUX_PLOT_FONTSIZE if is_linux else WIN_PLOT_FONTSIZE
        self.current_figure_size = LINUX_FIGURE_SIZE if is_linux else WIN_FIGURE_SIZE
        
        # Configure all styles
        style = ttk.Style()
        style.configure('Transform.TButton', font=(FONT_FAMILY, self.current_font_size), padding=self.current_padding)
        style.configure('Transform.Selected.TButton', font=(FONT_FAMILY, self.current_font_size, 'bold'), padding=self.current_padding)
        style.configure('Title.TLabel', font=(FONT_FAMILY, self.current_font_size, 'bold'))
        style.configure('Header.TLabel', font=(FONT_FAMILY, self.current_font_size, 'bold'))
        style.configure('Big.TLabel', font=(FONT_FAMILY, self.current_font_size))
        style.configure('Big.TButton', font=(FONT_FAMILY, self.current_font_size, 'bold'))
        style.configure('Big.TEntry', font=(FONT_FAMILY, self.current_font_size, 'bold'))
        
        # Configure modern switch style
        style.configure('Switch.TCheckbutton',
                       font=(FONT_FAMILY, self.current_font_size),
                       indicatorsize=25 if not is_linux else 20,
                       indicatormargin=5 if not is_linux else 3,
                       padding=5 if not is_linux else 3,
                       background='#ffffff',
                       foreground='#000000')

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create input frame
        input_frame = ttk.Frame(main_frame, padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matrix input at left
        matrix_frame = ttk.Frame(input_frame, padding="10")
        matrix_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20)
        
        ttk.Label(matrix_frame, text="Matrix G", style='Header.TLabel').grid(row=0, column=0, columnspan=2, padx=5, pady=2)
        
        # Create 2x2 matrix inputs
        self.matrix_entries = []
        for i in range(MATRIX_SIZE):
            row_entries = []
            for j in range(MATRIX_SIZE):
                entry = ttk.Entry(matrix_frame, width=self.current_entry_width, style='Big.TEntry', font=(FONT_FAMILY, self.current_font_size))
                entry.insert(0, str(self.config['matrix'][i][j]))
                entry.grid(row=i+1, column=j, padx=5, pady=2)
                entry.bind('<Return>', lambda e: self.transform())
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
        
        # Create coefficient frame
        coeff_frame = ttk.Frame(input_frame, padding="10")
        coeff_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create a single frame for all coefficients
        entry_frame = ttk.Frame(coeff_frame)
        entry_frame.grid(row=0, column=0, sticky='w')
        
        # Add order labels in a single row at the top
        for i in range(NUM_COEFFICIENTS):
            ttk.Label(entry_frame, text=f"Order {i+1}", style='Header.TLabel').grid(row=0, column=i+1, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
        
        # Add I and G labels on the left
        ttk.Label(entry_frame, text="I:", style='Header.TLabel').grid(row=1, column=0, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
        ttk.Label(entry_frame, text="G:", style='Header.TLabel').grid(row=2, column=0, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
        
        # Create entry fields for each order
        self.entries_a = []
        self.entries_b = []
        for i in range(NUM_COEFFICIENTS):
            # Entry for I coefficient (upper row)
            entry_a = ttk.Entry(entry_frame, width=self.current_entry_width, style='Big.TEntry', font=(FONT_FAMILY, self.current_font_size))
            entry_a.insert(0, str(self.config['coefficients_a'][i]))
            entry_a.grid(row=1, column=i+1, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
            entry_a.bind('<Return>', lambda e: self.transform())
            
            # Entry for G coefficient (lower row)
            entry_b = ttk.Entry(entry_frame, width=self.current_entry_width, style='Big.TEntry', font=(FONT_FAMILY, self.current_font_size))
            entry_b.insert(0, str(self.config['coefficients_b'][i]))
            entry_b.grid(row=2, column=i+1, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
            entry_b.bind('<Return>', lambda e: self.transform())
            
            # Store entries for later access
            self.entries_a.append(entry_a)
            self.entries_b.append(entry_b)
        
        # Create transform button frame
        button_frame = ttk.Frame(input_frame, padding="10")
        button_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=0)
        
        # Add title label
        title_label = ttk.Label(button_frame, text="Grid types", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 0))
        
        # Add spacing frame for vertical positioning
        spacing_frame = ttk.Frame(button_frame, height=0)
        spacing_frame.grid(row=1, column=0, columnspan=3)
        spacing_frame.grid_propagate(False)
        
        # Create grid of transform buttons
        buttons = [
            (GRID_TYPE_HV, self.transform),
            (GRID_TYPE_H, self.transform),
            (GRID_TYPE_V, self.transform),
            (GRID_TYPE_RC, self.transform),
            (GRID_TYPE_C, self.transform),
            (GRID_TYPE_R, self.transform)
        ]
        
        # Store buttons for later style updates
        self.transform_buttons = {}
        
        for i, (text, command) in enumerate(buttons):
            row = i // 3
            col = i % 3
            # Create a wrapper function to update button styles
            cmd = lambda t=text, c=command: self.update_transform_type(t, c)
            btn = ttk.Button(button_frame, text=text, command=cmd, style='Transform.TButton')
            btn.grid(row=row + 2, column=col, padx=4, pady=0, sticky=(tk.W, tk.E))
            self.transform_buttons[text] = btn
        
        # Add NUM_GRIDS input box
        num_grids_frame = ttk.Frame(button_frame)
        num_grids_frame.grid(row=2, column=3, padx=4, pady=(0, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(num_grids_frame, text="Number of lines: ", style='Big.TLabel').grid(row=0, column=0, padx=(10, 2), sticky=tk.E)
        self.num_grids_entry = ttk.Entry(num_grids_frame, width=3, style='Big.TEntry', font=(FONT_FAMILY, self.current_font_size))
        self.num_grids_entry.grid(row=0, column=1, sticky=tk.W)
        self.num_grids_entry.insert(0, str(self.config.get('num_grids', 4)))
        self.num_grids_entry.bind('<Return>', lambda e: self.update_num_grids())
        self.num_grids_entry.bind('<FocusOut>', lambda e: self.update_num_grids())
        
        # Add rotation toggle button
        self.rotation_enabled = tk.BooleanVar(value=self.config.get('rotation_enabled', True))
        rotation_btn = ttk.Checkbutton(button_frame, text="    45° Rotation ", 
                                     variable=self.rotation_enabled,
                                     command=self.transform,
                                     style='Switch.TCheckbutton')
        rotation_btn.grid(row=3, column=3, padx=19, pady=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Set initial selection from config
        self.current_transform_type = self.config['transform_type']
        self.update_button_styles()
        
        # Create matplotlib figure with platform-specific settings
        plt.rcParams.update({
            'font.size': self.current_plot_fontsize,
            'lines.linewidth': self.current_plot_linewidth,
            'figure.dpi': self.current_plot_dpi,
            'savefig.dpi': self.current_plot_dpi
        })
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=self.current_figure_size)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=3 if is_linux else 10, pady=3 if is_linux else 10)
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    

    def get_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current matrix values from the input fields.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - I (np.ndarray): Identity matrix of shape (2, 2)
                - G (np.ndarray): User-defined matrix of shape (2, 2)

        从输入框获取当前矩阵值。
        
        返回值:
            tuple[np.ndarray, np.ndarray]: 包含以下内容的元组:
                - I (np.ndarray): 单位矩阵，形状为(2, 2)
                - G (np.ndarray): 用户定义的矩阵，形状为(2, 2)
        """
        I = np.eye(MATRIX_SIZE)
        G = np.array([[float(entry.get()) for entry in row] for row in self.matrix_entries])
        G = G/np.sqrt(np.abs(np.linalg.det(G)))
        return I, G
    
    def get_coefficients(self) -> tuple[list[float], list[float]]:
        """
        Get the current coefficients from the input fields.
        
        Returns:
            tuple[list[float], list[float]]: A tuple containing:
                - coeffs_a (list[float]): Coefficients for I terms, length NUM_COEFFICIENTS
                - coeffs_b (list[float]): Coefficients for G terms, length NUM_COEFFICIENTS

        从输入框获取当前系数。
        
        返回值:
            tuple[list[float], list[float]]: 包含以下内容的元组:
                - coeffs_a (list[float]): I项的系数，长度为NUM_COEFFICIENTS
                - coeffs_b (list[float]): G项的系数，长度为NUM_COEFFICIENTS
        """
        coeffs_a = [float(entry.get()) for entry in self.entries_a]
        coeffs_b = [float(entry.get()) for entry in self.entries_b]
        return coeffs_a, coeffs_b
    
    def generate_points(self, grid_type: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate points for different grid patterns.
        
        Args:
            grid_type (str): Type of grid pattern to generate. Must be one of:
                - 'Horizontal+Vertical': Both horizontal and vertical lines
                - 'Horizontal': Only horizontal lines
                - 'Vertical': Only vertical lines
                - 'Circular+Radial': Both circular and radial lines
                - 'Circular': Only circular lines
                - 'Radial': Only radial lines
        
        Returns:
            list[tuple[np.ndarray, np.ndarray]]: List of point pairs, where each pair contains:
                - x coordinates array of shape (10000,)
                - y coordinates array of shape (10000,)

        生成不同网格模式的点集。
        
        参数:
            grid_type (str): 要生成的网格模式类型，必须是以下之一：
                - 'Horizontal+Vertical': 水平和垂直线
                - 'Horizontal': 仅水平线
                - 'Vertical': 仅垂直线
                - 'Circular+Radial': 圆形和径向线
                - 'Circular': 仅圆形线
                - 'Radial': 仅径向线
        
        返回值:
            list[tuple[np.ndarray, np.ndarray]]: 点对列表，每个点对包含：
                - x坐标数组，形状为(10000,)
                - y坐标数组，形状为(10000,)
        """
        points = []
        num_grids = self.get_num_grids()
        
        if 'Horizontal' in grid_type:
            for i in range(num_grids):
                y = -2 + i * 4 / (num_grids - 1)
                x = np.linspace(-2, 2, 10000)
                points.append((x, np.full(10000, y)))
        
        if 'Vertical' in grid_type:
            for i in range(num_grids):
                x = -2 + i * 4 / (num_grids - 1)
                y = np.linspace(-2, 2, 10000)
                points.append((np.full(10000, x), y))
        
        if 'Circular' in grid_type:
            theta = np.linspace(0, 2*np.pi, 10000)
            for i in range(num_grids):
                r = 0.1 + i * 1.9 / (num_grids - 1)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append((x, y))
        
        if 'Radial' in grid_type:
            r = np.linspace(0, 2, 10000)
            for i in range(num_grids):
                angle = i * 2*np.pi / num_grids
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append((x, y))
        
        return points

    def transform_points(self, points):
        """
        Transform points using the matrix polynomial transformation.
        
        This function applies the matrix polynomial transformation to each point:
        1. Computes the polynomial value using the current matrix G and coefficients
        2. Optionally applies a 45-degree rotation if enabled
        
        Args:
            points (list): List of (x, y) coordinate tuples to transform
            
        Returns:
            list: List of transformed (u, v) coordinate tuples
        """
        p_trans = []
        for p in points:
            x, y = p[0], p[1]
            I, G = self.get_matrix()
            coeffs_a, coeffs_b = self.get_coefficients()

            u, v = compute_polynomial(x, y, I, G, coeffs_a, coeffs_b)
            
            # Apply final transformation
            if self.rotation_enabled.get():
                # Apply 45-degree rotation: x' = (P1-P2)/√2, y' = (P1+P2)/√2
                u1 = (u - v) / np.sqrt(2)
                v1 = (u + v) / np.sqrt(2)
            else:
                u1, v1 = u, v
            p_trans.append((u1, v1))
        
        return p_trans     
    
    
    def plot_transformation(self, input_points, title_suffix=""):
        """
        Plot the transformation of the given points.
        
        Args:
            input_points (list): List of (x, y) points
            title_suffix (str): Suffix for the plot title
        """
        # Clear the plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Track maximum range for plot limits
        input_max_range = 0
        transformed_max_range = 0
        
        # Get line style based on grid type
        def get_line_style(idx, title_suffix):
            if GRID_TYPE_HV in title_suffix:
                return LINE_STYLE_SOLID if idx < self.get_num_grids() else LINE_STYLE_DOTTED
            elif GRID_TYPE_RC in title_suffix:
                return LINE_STYLE_SOLID if idx < self.get_num_grids() else LINE_STYLE_DOTTED
            elif GRID_TYPE_H in title_suffix or GRID_TYPE_C in title_suffix:
                return LINE_STYLE_SOLID
            else:
                return LINE_STYLE_DOTTED
        
        # Get color and alpha values based on grid type and index
        def get_color_and_alpha(idx, title_suffix):
            num_grids = self.get_num_grids()

            # Determine effective index based on grid type
            if GRID_TYPE_HV in title_suffix or GRID_TYPE_RC in title_suffix:
                idx_eff = idx % num_grids
            else:
                idx_eff = idx

            if num_grids <= 10:
                # Use matplotlib default colors C0-C9 with high alpha for few lines
                return f'C{idx_eff}', ALPHA_FEW_LINES
            else:
                # Use viridis colormap with lower alpha for many lines to prevent visual clutter
                import matplotlib.cm as cm
                return cm.viridis(idx_eff/(num_grids)), ALPHA_MANY_LINES
        
        # Plot input points and calculate input range
        for idx, (x, y) in enumerate(input_points):
            style = get_line_style(idx, title_suffix)
            color, alpha = get_color_and_alpha(idx, title_suffix)
            self.ax1.plot(x, y, style, color=color, linewidth=self.current_plot_linewidth, alpha=alpha)
            input_max_range = max(input_max_range, np.max(np.abs([x, y])))
        
        # Plot transformed points and calculate transformed range
        transformed_points = self.transform_points(input_points)
        for idx, (x, y) in enumerate(transformed_points):
            style = get_line_style(idx, title_suffix)
            color, alpha = get_color_and_alpha(idx, title_suffix)
            self.ax2.plot(x, y, style, color=color, linewidth=self.current_plot_linewidth, alpha=alpha)
            transformed_max_range = max(transformed_max_range, np.max(np.abs([x, y])))
        
        # Set plot properties with separate ranges
        input_plot_range = input_max_range * PLOT_MARGIN
        transformed_plot_range = transformed_max_range * PLOT_MARGIN
        
        # Set properties for input plot (ax1)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlim(-input_plot_range, input_plot_range)
        self.ax1.set_ylim(-input_plot_range, input_plot_range)
        self.ax1.axhline(y=0, color='gray', linewidth=0.5)
        self.ax1.axvline(x=0, color='gray', linewidth=0.5)
        # Set 5 ticks on each axis
        self.ax1.set_xticks(np.linspace(-input_plot_range, input_plot_range, 5))
        self.ax1.set_yticks(np.linspace(-input_plot_range, input_plot_range, 5))
        
        # Set properties for transformed plot (ax2)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(-transformed_plot_range, transformed_plot_range)
        self.ax2.set_ylim(-transformed_plot_range, transformed_plot_range)
        self.ax2.axhline(y=0, color='gray', linewidth=0.5)
        self.ax2.axvline(x=0, color='gray', linewidth=0.5)
        # Set 5 ticks on each axis
        self.ax2.set_xticks(np.linspace(-transformed_plot_range, transformed_plot_range, 5))
        self.ax2.set_yticks(np.linspace(-transformed_plot_range, transformed_plot_range, 5))
        
        self.ax1.set_title(f'$x$-$y$ plane', fontsize=self.current_plot_fontsize)
        self.ax2.set_title(f'$u$-$v$ plane', fontsize=self.current_plot_fontsize)
        
        # Display polynomial equation
        coeffs_a, coeffs_b = self.get_coefficients()
        terms = []
        for i in range(NUM_COEFFICIENTS):
            if coeffs_a[i] != 0 or coeffs_b[i] != 0:
                coeff_str = []
                if coeffs_a[i] != 0:
                    coeff_str.append(f"{coeffs_a[i]:.2f}\\mathbf{{I}}")
                if coeffs_b[i] != 0:
                    sign = "+" if coeffs_b[i] > 0 and coeff_str else ""
                    coeff_str.append(f"{sign}{coeffs_b[i]:.2f}\\mathbf{{G}}")
                
                term = f"({' '.join(coeff_str)})(x\\mathbf{{I}}+y\\mathbf{{G}})"
                if i > 0:
                    term += f"^{i+1}"
                terms.append(term)
        
        poly_str = "$" + "+".join(terms) if terms else "$0"
        poly_str += "$"
        
        # Clear old text if it exists
        if hasattr(self, 'poly_text'):
            self.poly_text.remove()
        self.poly_text = self.fig.text(0.5, 0.02, f"$u\\mathbf{{I}}+v\\mathbf{{G}}$ = {poly_str}", 
                                     fontsize=self.current_plot_fontsize, ha='center')
        
        # Adjust subplot spacing
        self.fig.subplots_adjust(bottom=0.17, wspace=0.2, top=0.92, left=0.08, right=0.98)  # Increase horizontal spacing between subplots
        
        # Update canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
 
    def transform(self):
        """
        Transform the current grid pattern.
        """
        self.plot_transformation(self.generate_points(self.current_transform_type))
    
    def update_transform_type(self, button_text, command):
        """
        Update the current transform type and button styles.
        
        Args:
            button_text (str): Text of the selected button
            command (function): Command to execute when the button is clicked
        """
        self.current_transform_type = button_text
        self.update_button_styles()
        command()
    
    def update_button_styles(self):
        """Update the styles of the transform buttons."""
        for text, button in self.transform_buttons.items():
            if text == self.current_transform_type:
                button.configure(style='Transform.Selected.TButton')
            else:
                button.configure(style='Transform.TButton')
    
    def update_num_grids(self, *args):
        """
        Update the number of grid lines.
        
        Args:
            *args: Additional arguments (not used)
        """
        try:
            value = int(self.num_grids_entry.get())
            if value < 2:
                value = 2
            elif value > 250:
                value = 250
            self.num_grids_entry.delete(0, tk.END)
            self.num_grids_entry.insert(0, str(value))
            self.transform()
        except ValueError:
            # Reset to default if invalid input
            self.num_grids_entry.delete(0, tk.END)
            self.num_grids_entry.insert(0, str(self.config.get('num_grids', 4)))
    
    def get_num_grids(self) -> int:
        """
        Get the current number of grid lines.
        
        Returns:
            int: Number of grid lines, clamped between 2 and 250
        """
        try:
            return max(2, min(250, int(self.num_grids_entry.get())))
        except ValueError:
            return 4
    
    def load_config(self):
        """
        Load saved configuration from JSON file or use default values.
        
        Returns:
            dict: Loaded configuration
        """
        default_config = {
            'coefficients_a': [0.0] * NUM_COEFFICIENTS,
            'coefficients_b': [0.0] * NUM_COEFFICIENTS,
            'matrix': [[1.0, 0.0], [0.0, 1.0]],
            'transform_type': GRID_TYPE_HV,
            'rotation_enabled': True,
            'num_grids': 4
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Ensure all required keys exist
                    for key in default_config:
                        if key not in loaded_config:
                            loaded_config[key] = default_config[key]
                    return loaded_config
        except Exception as e:
            print(f"Error loading configuration: {e}")
        
        return default_config
    
    def save_config(self):
        """
        Save current configuration to JSON file.
        """
        config = {
            'coefficients_a': [float(entry.get()) for entry in self.entries_a],
            'coefficients_b': [float(entry.get()) for entry in self.entries_b],
            'matrix': [[float(entry.get()) for entry in row] for row in self.matrix_entries],
            'transform_type': self.current_transform_type,
            'rotation_enabled': self.rotation_enabled.get(),
            'num_grids': self.get_num_grids()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def on_closing(self):
        """
        Clean up resources when closing the application.
        """
        try:
            self.save_config()
            # Clean up matplotlib resources
            plt.close(self.fig)
            self.canvas.get_tk_widget().destroy()
        finally:
            self.root.quit()
            self.root.destroy()   
    # The rest of the methods are identical to the original class
    # [Other methods omitted for brevity - identical to original]

if __name__ == "__main__":
    root = tk.Tk()
    
    # Platform-specific window setup
    if platform.system() == 'Linux':
        # Linux-specific DPI and window settings
        # Reset to baseline scaling
        root.tk.call('tk', 'scaling', 1.0)
        # Get screen DPI
        dpi = root.winfo_fpixels('1i')
        # Calculate scaling factor (96 is the base DPI)
        scaling = 96.0 / dpi if dpi > 0 else 1.0
        # Apply Linux-specific scaling
        root.tk.call('tk', 'scaling', scaling)
        # Linux-specific window maximization
        root.attributes('-zoomed', True)
    else:
        # Windows-specific window maximization
        root.state('zoomed')
    
    app = MatrixPolynomialAppOptimized(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
