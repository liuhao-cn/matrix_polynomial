"""
Matrix Polynomial Visualization Tool

This module provides a graphical interface for visualizing matrix polynomial transformations.
The tool allows users to:
- Input a 2x2 matrix G and polynomial coefficients
- Choose from various grid patterns for visualization
- See real-time transformation results
- Save plots as PDF files
- Scan through coefficient values for animation

The module handles:
- User interface and interaction
- Grid pattern generation for visualization
- Real-time plotting and updates
- Configuration management
- PDF export functionality

矩阵多项式可视化工具

本模块提供了矩阵多项式变换的图形界面。
该工具允许用户：
- 输入 2x2 矩阵 G 和多项式系数
- 选择不同的网格模式进行可视化
- 实时查看变换结果
- 将图形保存为 PDF 文件
- 通过系数扫描制作动画

本模块负责：
- 用户界面和交互
- 生成可视化用的网格模式
- 实时绘图和更新
- 配置管理
- PDF 导出功能
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import os
import sys
import platform
from tkinter import filedialog
import datetime
from pathlib import Path
from matrix_polynomial_math import decompose_matrix, compute_polynomial

# Grid Generation Constants
GRID_RANGE = (-2, 2)      # Range for x and y coordinates
NUM_POINTS = 10000        # Number of points for line generation
MIN_CIRCLE_RADIUS = 0.1   # Minimum radius for circular grids
MAX_CIRCLE_RADIUS = 2.0   # Maximum radius for circular grids
ALPHA_FEW_LINES = 1.0     # Alpha value for few grid lines
ALPHA_MANY_LINES = 0.6    # Alpha value for many grid lines

# GUI Constants
# Windows-specific GUI parameters
WIN_FONT_SIZE = 23
WIN_BUTTON_PADDING = (10, 0)
WIN_ENTRY_WIDTH = 6
WIN_PLOT_DPI = 600
WIN_PLOT_LINEWIDTH = 0.7
WIN_PLOT_FONTSIZE = 5
WIN_FIGURE_SIZE = (4, 2)

# Linux-specific GUI parameters
LINUX_FONT_SIZE = 11
LINUX_BUTTON_PADDING = (5, 0)
LINUX_ENTRY_WIDTH = 5
LINUX_PLOT_DPI = 150
LINUX_PLOT_LINEWIDTH = 1.0
LINUX_PLOT_FONTSIZE = 8
LINUX_FIGURE_SIZE = (8, 4)

# Common parameters
FONT_FAMILY = 'Arial'
MATRIX_SIZE = 2
NUM_COEFFICIENTS = 6
COEFFICIENTS_PER_FRAME = 2
PLOT_MARGIN = 1.1        # Plot range margin (10% extra)

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
LINE_STYLE_DOTTED = '-'   # Changed from ':' to '--' for dashed style

# Output Constants
OUTPUT_DIR = "output"     # Output directory for PDF files
INFO_DISPLAY_TIME = 15000 # Time (ms) to display info message

# Scanning Constants
SCAN_STEP = 0.02         # Step size for coefficient scanning
SCAN_RANGE = (-1.0, 1.0) # Range for scanning
SCAN_DELAY = 100         # Delay between scans (ms)

class MatrixPolynomialApp:
    """
    Matrix Polynomial visualization tool.
    
    This class provides a graphical interface for visualizing matrix polynomial 
    transformations. It includes:
    - Interactive matrix and coefficient input
    - Multiple grid pattern options
    - Real-time visualization
    - PDF export functionality
    - Coefficient scanning animation
    
    矩阵多项式可视化工具。
    
    该类提供了用于可视化矩阵多项式变换的图形界面。包括：
    - 交互式矩阵和系数输入
    - 多种网格模式选项
    - 实时可视化
    - PDF导出功能
    - 系数扫描动画
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
        self.root.title("Matrix Polynomial")
        
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

        # Add info message style
        style.configure('Info.TLabel', 
                       font=(FONT_FAMILY, self.current_font_size),
                       foreground='green')

        # Add scan button style with bold font
        style.configure('Scan.TButton',
                       font=(FONT_FAMILY, self.current_font_size, 'bold'))  # 添加粗体样式

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
                entry.bind('<Return>', lambda e: self.update_plot())
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
        
        # Create coefficient frame
        coeff_frame = ttk.Frame(input_frame, padding="10")
        coeff_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create a single frame for all coefficients
        entry_frame = ttk.Frame(coeff_frame)
        entry_frame.grid(row=0, column=0, sticky='w')
        
        # Replace Order labels with Scan buttons
        self.scan_buttons = []
        for i in range(NUM_COEFFICIENTS):
            scan_button = ttk.Button(entry_frame,
                                   text=f"Scan {i+1}",
                                   command=lambda x=i: self.start_scan(x),
                                   style='Scan.TButton',  # 使用新的粗体样式
                                   width=self.current_entry_width)
            scan_button.grid(row=0, 
                            column=i+1, 
                            padx=3 if is_linux else 5, 
                            pady=1 if is_linux else 2,
                            sticky='ew')
            self.scan_buttons.append(scan_button)
        
        # Add scanning state variables
        self.current_scan = None  # Track current scanning process
        self.scanning = False     # Flag to indicate if scanning is active
        
        # Add I and G labels on the left
        ttk.Label(entry_frame, text="I:", style='Header.TLabel').grid(
            row=1, column=0, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
        ttk.Label(entry_frame, text="G:", style='Header.TLabel').grid(
            row=2, column=0, padx=3 if is_linux else 5, pady=1 if is_linux else 2)
        
        # Create entry fields for each order
        self.entries_a = []
        self.entries_b = []
        for i in range(NUM_COEFFICIENTS):
            # Entry for I coefficient (upper row)
            entry_a = ttk.Entry(entry_frame, 
                               width=self.current_entry_width,
                               style='Big.TEntry',
                               font=(FONT_FAMILY, self.current_font_size))
            entry_a.insert(0, str(self.config['coefficients_a'][i]))
            entry_a.grid(row=1, column=i+1, padx=3 if is_linux else 5, pady=1 if is_linux else 2, sticky='ew')
            entry_a.bind('<Return>', lambda e: self.update_plot())
            
            # Entry for G coefficient (lower row)
            entry_b = ttk.Entry(entry_frame,
                               width=self.current_entry_width,
                               style='Big.TEntry',
                               font=(FONT_FAMILY, self.current_font_size))
            entry_b.insert(0, str(self.config['coefficients_b'][i]))
            entry_b.grid(row=2, column=i+1, padx=3 if is_linux else 5, pady=1 if is_linux else 2, sticky='ew')
            entry_b.bind('<Return>', lambda e: self.update_plot())
            
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
            (GRID_TYPE_HV, self.update_plot),
            (GRID_TYPE_H, self.update_plot),
            (GRID_TYPE_V, self.update_plot),
            (GRID_TYPE_RC, self.update_plot),
            (GRID_TYPE_C, self.update_plot),
            (GRID_TYPE_R, self.update_plot)
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
        
        # Create a frame for both entries
        control_frame = ttk.Frame(button_frame)
        control_frame.grid(row=2, column=3, rowspan=2, padx=4, pady=(0, 0), sticky=(tk.W, tk.E))
        control_frame.grid_columnconfigure(1, minsize=50)  # 设置第二列（输入框列）的最小宽度

        # Number of lines entry
        ttk.Label(control_frame, text="Number of lines:", style='Big.TLabel').grid(
            row=0, column=0, padx=(10, 2), sticky=tk.E)
        self.num_grids_entry = ttk.Entry(control_frame, width=3, style='Big.TEntry',
                                        font=(FONT_FAMILY, self.current_font_size))
        self.num_grids_entry.grid(row=0, column=1, sticky=tk.E)
        self.num_grids_entry.insert(0, str(self.config.get('num_grids', 4)))
        self.num_grids_entry.bind('<Return>', lambda e: self.update_num_grids())
        self.num_grids_entry.bind('<FocusOut>', lambda e: self.update_num_grids())

        # Plot range entry
        ttk.Label(control_frame, text="Plot Range: ±", style='Big.TLabel').grid(
            row=1, column=0, padx=(10, 2), sticky=tk.E)
        self.range_entry = ttk.Entry(control_frame, width=3, style='Big.TEntry',
                                    font=(FONT_FAMILY, self.current_font_size))
        self.range_entry.grid(row=1, column=1, sticky=tk.E)
        self.range_entry.insert(0, str(abs(GRID_RANGE[0])))
        self.range_entry.bind('<Return>', lambda e: self.update_range())
        self.range_entry.bind('<FocusOut>', lambda e: self.update_range())
        
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
        
        # Add save button
        save_button = ttk.Button(button_frame, 
                                text="Save\nPDF", 
                                command=self.save_plot_as_pdf,
                                style='Big.TButton',
                                width=7)  # 设置较小的固定宽度
        save_button.grid(row=2, 
                        column=4, 
                        rowspan=2, 
                        padx=2,  # 减小水平padding
                        pady=1, 
                        sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    

    def read_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current matrix values from the input fields.
        This function:
        1. Creates a 2x2 identity matrix I
        2. Reads the user-defined matrix G from input fields
        3. Normalizes G by its determinant
        
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - I (np.ndarray): Identity matrix of shape (2, 2)
                - G (np.ndarray): User-defined matrix of shape (2, 2)

        从输入框获取当前矩阵值。
        此函数：
        1. 创建 2x2 单位矩阵 I
        2. 从输入框读取用户定义的矩阵 G
        3. 通过行列式对 G 进行归一化
        
        返回值：
            tuple[np.ndarray, np.ndarray]：包含以下内容的元组：
                - I (np.ndarray)：单位矩阵，形状为 (2, 2)
                - G (np.ndarray)：用户定义的矩阵，形状为 (2, 2)
        """
        I = np.eye(MATRIX_SIZE)
        G = np.array([[float(entry.get()) for entry in row] for row in self.matrix_entries])
        G = G/np.sqrt(np.abs(np.linalg.det(G)))
        return I, G
    
    def read_coefficients(self) -> tuple[list[float], list[float]]:
        """
        Get the current coefficients from the input fields.
        This function reads both I and G coefficients from their respective input fields.
        
        Returns:
            tuple[list[float], list[float]]: A tuple containing:
                - coeffs_a (list[float]): Coefficients for I terms
                - coeffs_b (list[float]): Coefficients for G terms

        从输入框获取当前系数。
        此函数从各自的输入框读取 I 和 G 的系数。
        
        返回值：
            tuple[list[float], list[float]]：包含以下内容的元组：
                - coeffs_a (list[float])：I 项的系数
                - coeffs_b (list[float])：G 项的系数
        """
        coeffs_a = [float(entry.get()) for entry in self.entries_a]
        coeffs_b = [float(entry.get()) for entry in self.entries_b]
        return coeffs_a, coeffs_b
    
    def generate_points(self, grid_type: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate points for different grid patterns in the complex plane.
        Supports four types of grids:
        - Horizontal lines
        - Vertical lines
        - Circular lines (concentric circles)
        - Radial lines (rays from origin)
        These can be combined to create composite patterns.
        
        Args:
            grid_type (str): Type of grid pattern to generate
        
        Returns:
            list[tuple[np.ndarray, np.ndarray]]: List of point pairs (x, y)

        在复平面上生成不同类型的网格点。
        支持四种网格类型：
        - 水平线
        - 垂直线
        - 圆形线（同心圆）
        - 径向线（从原点发出的射线）
        这些可以组合形成复合模式。
        
        参数：
            grid_type (str)：要生成的网格类型
        
        返回值：
            list[tuple[np.ndarray, np.ndarray]]：点对 (x, y) 的列表
        """
        points = []
        num_grids = self.get_num_grids()

        if 'Horizontal' in grid_type:
            for i in range(num_grids):
                y = GRID_RANGE[0] + i * (GRID_RANGE[1] - GRID_RANGE[0]) / (num_grids - 1)
                x = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
                points.append((x, np.full(NUM_POINTS, y)))

        if 'Vertical' in grid_type:
            for i in range(num_grids):
                x = GRID_RANGE[0] + i * (GRID_RANGE[1] - GRID_RANGE[0]) / (num_grids - 1)
                y = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
                points.append((np.full(NUM_POINTS, x), y))

        if 'Circular' in grid_type:
            theta = np.linspace(0, 2*np.pi, NUM_POINTS)
            max_radius = abs(GRID_RANGE[1])
            for i in range(num_grids):
                r = MIN_CIRCLE_RADIUS + i * (max_radius - MIN_CIRCLE_RADIUS) / (num_grids - 1)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append((x, y))

        if 'Radial' in grid_type:
            r = np.linspace(0, abs(GRID_RANGE[1]), NUM_POINTS)
            for i in range(num_grids):
                angle = i * 2*np.pi / num_grids
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append((x, y))

        return points

    def transform(self, points: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Transform points using the matrix polynomial transformation.
        This function:
        1. Reads the current matrix G and coefficients from GUI
        2. Applies the polynomial transformation to each set of points
        3. Returns the transformed coordinates
        
        Args:
            points (list[tuple[np.ndarray, np.ndarray]]): List of point pairs to transform,
                where each pair contains x and y coordinate arrays
            
        Returns:
            list[tuple[np.ndarray, np.ndarray]]: List of transformed point pairs (u, v)

        使用矩阵多项式变换对点进行变换。
        此函数：
        1. 从 GUI 读取当前矩阵 G 和系数
        2. 对每组点应用多项式变换
        3. 返回变换后的坐标
        
        参数：
            points (list[tuple[np.ndarray, np.ndarray]])：要变换的点对列表，
                每个点对包含 x 和 y 坐标数组
            
        返回值：
            list[tuple[np.ndarray, np.ndarray]]：变换后的点对 (u, v) 列表
        """
        transformed_points = []
        I, G = self.read_matrix()
        coeffs_a, coeffs_b = self.read_coefficients()

        for p in points:
            x, y = p[0], p[1]
            u, v = compute_polynomial(x, y, I, G, coeffs_a, coeffs_b)
            transformed_points.append((u, v))

        return transformed_points

    def plot_transformation(self, 
                          input_points: list[tuple[np.ndarray, np.ndarray]],
                          transformed_points: list[tuple[np.ndarray, np.ndarray]],
                          title_suffix: str = "") -> None:
        """
        Plot the transformation of input points in both input and output planes.
        This function:
        1. Clears both input and output plots
        2. Plots original points in the left subplot
        3. Plots transformed points in the right subplot
        4. Adjusts plot ranges and styles automatically
        5. Displays the current polynomial equation
        
        Args:
            input_points (list[tuple[np.ndarray, np.ndarray]]): Original point pairs
            transformed_points (list[tuple[np.ndarray, np.ndarray]]): Transformed point pairs
            title_suffix (str, optional): Additional text to append to plot titles

        在输入平面和输出平面上绘制点的变换。
        此函数：
        1. 清除输入和输出图形
        2. 在左侧子图中绘制原始点
        3. 在右侧子图中绘制变换后的点
        4. 自动调整绘图范围和样式
        5. 显示当前多项式方程
        
        参数：
            input_points (list[tuple[np.ndarray, np.ndarray]])：原始点对
            transformed_points (list[tuple[np.ndarray, np.ndarray]])：变换后的点对
            title_suffix (str, optional)：要附加到图表标题的额外文本
        """
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
            if GRID_TYPE_HV in title_suffix or GRID_TYPE_RC in title_suffix:
                idx_eff = idx % num_grids
            else:
                idx_eff = idx
            
            if num_grids <= 10:
                return f'C{idx_eff}', ALPHA_FEW_LINES
            else:
                import matplotlib.cm as cm
                return cm.viridis(idx_eff/(num_grids)), ALPHA_MANY_LINES
        
        # Plot input points and calculate input range
        for idx, (x, y) in enumerate(input_points):
            style = get_line_style(idx, title_suffix)
            color, alpha = get_color_and_alpha(idx, title_suffix)
            self.ax1.plot(x, y, style, color=color, linewidth=self.current_plot_linewidth, alpha=alpha)
            input_max_range = max(input_max_range, np.max(np.abs([x, y])))
        
        # Plot transformed points and calculate transformed range
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
        self.ax1.axhline(y=0, color='gray', linewidth=0.2)
        self.ax1.axvline(x=0, color='gray', linewidth=0.2)
        # Set 5 ticks on each axis
        self.ax1.set_xticks(np.linspace(-input_plot_range, input_plot_range, 5))
        self.ax1.set_yticks(np.linspace(-input_plot_range, input_plot_range, 5))
        
        # Set properties for transformed plot (ax2)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(-transformed_plot_range, transformed_plot_range)
        self.ax2.set_ylim(-transformed_plot_range, transformed_plot_range)
        self.ax2.axhline(y=0, color='gray', linewidth=0.2)
        self.ax2.axvline(x=0, color='gray', linewidth=0.2)
        # Set 5 ticks on each axis
        self.ax2.set_xticks(np.linspace(-transformed_plot_range, transformed_plot_range, 5))
        self.ax2.set_yticks(np.linspace(-transformed_plot_range, transformed_plot_range, 5))
        
        self.ax1.set_title(f'$x$-$y$ plane', fontsize=self.current_plot_fontsize)
        self.ax2.set_title(f'$u$-$v$ plane', fontsize=self.current_plot_fontsize)
        
        # Display polynomial equation
        coeffs_a, coeffs_b = self.read_coefficients()
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
    
 
    def update_transform_type(self, button_text: str, command: callable) -> None:
        """
        Update the current transform type and button styles.
        This function:
        1. Updates the current transform type
        2. Updates button styles to highlight the selected type
        3. Triggers a plot update
        
        Args:
            button_text (str): Text of the selected button
            command (callable): Command to execute when the button is clicked

        更新当前变换类型和按钮样式。
        此函数：
        1. 更新当前变换类型
        2. 更新按钮样式以突出显示所选类型
        3. 触发图形更新
        
        参数：
            button_text (str)：选中按钮的文本
            command (callable)：点击按钮时执行的命令
        """
        self.current_transform_type = button_text
        self.update_button_styles()
        self.update_plot()

    def update_button_styles(self) -> None:
        """
        Update the styles of the transform buttons.
        Highlights the currently selected transform type button and
        sets all other buttons to their default style.

        更新变换按钮的样式。
        突出显示当前选中的变换类型按钮，
        并将所有其他按钮设置为默认样式。
        """
        for text, button in self.transform_buttons.items():
            if text == self.current_transform_type:
                button.configure(style='Transform.Selected.TButton')
            else:
                button.configure(style='Transform.TButton')

    def update_num_grids(self, *args) -> None:
        """
        Update the number of grid lines.
        This function:
        1. Reads the value from the entry field
        2. Clamps it between 2 and 250
        3. Updates the display and triggers a plot update
        
        Args:
            *args: Additional arguments (not used)

        更新网格线数量。
        此函数：
        1. 从输入框读取值
        2. 将其限制在 2 到 250 之间
        3. 更新显示并触发图形更新
        
        参数：
            *args：额外的参数（未使用）
        """
        try:
            value = int(self.num_grids_entry.get())
            if value < 2:
                value = 2
            elif value > 250:
                value = 250
            self.num_grids_entry.delete(0, tk.END)
            self.num_grids_entry.insert(0, str(value))
            self.update_plot()
        except ValueError:
            # Reset to default if invalid input
            self.num_grids_entry.delete(0, tk.END)
            self.num_grids_entry.insert(0, str(self.config.get('num_grids', 4)))

    def update_plot(self) -> None:
        """
        Update the plot with current grid pattern and its transformation.
        This function:
        1. Generates grid points based on current settings
        2. Transforms these points using the current polynomial
        3. Updates the plot with both original and transformed points
        
        更新当前网格模式及其变换的图形。
        此函数：
        1. 基于当前设置生成网格点
        2. 使用当前多项式变换这些点
        3. 使用原始点和变换后的点更新图形
        """
        input_points = self.generate_points(self.current_transform_type)
        transformed_points = self.transform(input_points)
        self.plot_transformation(input_points, transformed_points)
    
    def get_num_grids(self) -> int:
        """
        Get the current number of grid lines.
        Returns a value clamped between 2 and 250, with a default of 4
        if the current value is invalid.
        
        Returns:
            int: Number of grid lines, clamped between 2 and 250

        获取当前网格线数量。
        返回一个在 2 到 250 之间的值，如果当前值无效则返回默认值 4。
        
        返回值：
            int：网格线数量，限制在 2 到 250 之间
        """
        try:
            return max(2, min(250, int(self.num_grids_entry.get())))
        except ValueError:
            return 4
    
    def load_config(self) -> dict:
        """
        Load saved configuration from JSON file or use defaults.
        This function:
        1. Attempts to read the configuration file
        2. Validates all required keys exist
        3. Returns default values if file is missing or invalid
        
        Returns:
            dict: Configuration dictionary containing:
                - coefficients_a (list[float]): Coefficients for I terms
                - coefficients_b (list[float]): Coefficients for G terms
                - matrix (list[list[float]]): 2x2 matrix G
                - transform_type (str): Current grid pattern type
                - num_grids (int): Number of grid lines

        从JSON文件加载保存的配置或使用默认值。
        此函数：
        1. 尝试读取配置文件
        2. 验证所有必需的键是否存在
        3. 如果文件缺失或无效则返回默认值
        
        返回值：
            dict：配置字典，包含：
                - coefficients_a (list[float])：I项的系数
                - coefficients_b (list[float])：G项的系数
                - matrix (list[list[float]])：2x2矩阵G
                - transform_type (str)：当前网格模式类型
                - num_grids (int)：网格线数量
        """
        default_config = {
            'coefficients_a': [0.0] * NUM_COEFFICIENTS,
            'coefficients_b': [0.0] * NUM_COEFFICIENTS,
            'matrix': [[1.0, 0.0], [0.0, 1.0]],
            'transform_type': GRID_TYPE_HV,
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

    def save_config(self) -> None:
        """
        Save current configuration to JSON file.
        This function saves:
        1. Current matrix G values
        2. All polynomial coefficients
        3. Selected transform type
        4. Number of grid lines
        5. Current plot range

        将当前配置保存到JSON文件。
        此函数保存：
        1. 当前矩阵 G 的值
        2. 所有多项式系数
        3. 选定的变换类型
        4. 网格线数量
        5. 当前绘图范围
        """
        config = {
            'coefficients_a': [float(entry.get()) for entry in self.entries_a],
            'coefficients_b': [float(entry.get()) for entry in self.entries_b],
            'matrix': [[float(entry.get()) for entry in row] for row in self.matrix_entries],
            'transform_type': self.current_transform_type,
            'num_grids': self.get_num_grids()
        }

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def on_closing(self) -> None:
        """
        Clean up resources when closing the application.
        This function:
        1. Saves the current configuration
        2. Cleans up matplotlib resources
        3. Destroys the GUI window

        清理应用程序关闭时的资源。
        此函数：
        1. 保存当前配置
        2. 清理matplotlib资源
        3. 销毁GUI窗口
        """
        try:
            self.save_config()
            # Clean up matplotlib resources
            plt.close(self.fig)
            self.canvas.get_tk_widget().destroy()
        finally:
            self.root.quit()
            self.root.destroy()

    def update_range(self, *args) -> None:
        """
        Update the grid range based on user input.
        
        The range will be set to [-value, value] where value is the user input,
        clamped between 0.1 and 100.

        更新网格范围，基于用户输入。
        
        范围将被设置为 [-value, value]，其中 value 是用户输入值，
        限制在 0.1 到 100 之间。
        """
        try:
            value = float(self.range_entry.get())
            value = max(0.1, min(100, value))
            self.range_entry.delete(0, tk.END)
            self.range_entry.insert(0, str(value))
            global GRID_RANGE
            GRID_RANGE = (-value, value)
            self.update_plot()
        except ValueError:
            self.range_entry.delete(0, tk.END)
            self.range_entry.insert(0, str(abs(GRID_RANGE[0])))

    def show_info(self, message: str) -> None:
        """
        Show information message in the UI and console.
        
        Args:
            message (str): Message to display

        在 UI 和控制台显示信息。
        
        参数:
            message (str): 要显示的信息
        """
        # Print to console
        print(message)
        
        # Show in UI
        if hasattr(self, 'info_label'):
            self.info_label.destroy()
        
        self.info_label = ttk.Label(self.root, text=message, style='Info.TLabel')
        self.info_label.grid(row=2, column=0, padx=5, pady=5)
        
        # Schedule label removal
        self.root.after(INFO_DISPLAY_TIME, self.clear_info)

    def clear_info(self) -> None:
        """
        Clear the information message from UI.
        
        从 UI 中清除信息提示。
        """
        if hasattr(self, 'info_label'):
            self.info_label.destroy()
            del self.info_label

    def save_plot_as_pdf(self) -> None:
        """
        Save the current plot as a PDF file in the output directory.
        The default filename includes the current date and time.

        将当前图形保存为 PDF 文件到输出目录。
        默认文件名包含当前日期和时间。
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(os.path.dirname(os.path.abspath(__file__))) / OUTPUT_DIR
            output_path.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_path / f"matrix_polynomial_{timestamp}.pdf"
            
            # Temporarily adjust figure size for better PDF quality
            original_size = self.fig.get_size_inches()
            self.fig.set_size_inches(12, 6)
            
            # Save with high DPI
            self.fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            
            # Restore original figure size
            self.fig.set_size_inches(*original_size)
            
            # Update canvas
            self.canvas.draw()
            
            # Show success message
            self.show_info(f"PDF saved: {filename}")
            
        except Exception as e:
            error_msg = f"Error saving PDF: {e}"
            print(error_msg)
            self.show_info(error_msg)

    def start_scan(self, order: int) -> None:
        """
        Start scanning coefficients for the specified order.
        This function:
        1. Stops any ongoing scan
        2. Initializes new scan parameters
        3. Updates button appearance
        4. Starts the scanning process
        
        Args:
            order (int): The order to scan (0-5)

        开始扫描指定阶数的系数。
        此函数：
        1. 停止任何正在进行的扫描
        2. 初始化新的扫描参数
        3. 更新按钮外观
        4. 启动扫描过程
        
        参数：
            order (int)：要扫描的阶数 (0-5)
        """
        if self.scanning:
            self.stop_scan()
        
        self.scanning = True
        self.current_scan = {
            'order': order,
            'value': SCAN_RANGE[0],
            'original_a': float(self.entries_a[order].get()),
            'original_b': float(self.entries_b[order].get())
        }
        
        # Update button text and command
        self.scan_buttons[order].configure(
            text="Stop",
            command=self.stop_scan
        )
        
        # Start scanning
        self.continue_scan()

    def stop_scan(self) -> None:
        """
        Stop the current scanning process and restore original values.
        This function:
        1. Restores original coefficient values
        2. Resets button appearance
        3. Updates the plot
        4. Clears scanning state

        停止当前扫描过程并恢复原始值。
        此函数：
        1. 恢复原始系数值
        2. 重置按钮外观
        3. 更新图形
        4. 清除扫描状态
        """
        if self.scanning and self.current_scan:
            order = self.current_scan['order']
            
            # Restore original values
            self.entries_a[order].delete(0, tk.END)
            self.entries_b[order].delete(0, tk.END)
            self.entries_a[order].insert(0, str(self.current_scan['original_a']))
            self.entries_b[order].insert(0, str(self.current_scan['original_b']))
            
            # Reset button text and command
            self.scan_buttons[order].configure(
                text=f"Scan {order+1}",
                command=lambda x=order: self.start_scan(x)
            )
            
            # Reset scanning state
            self.scanning = False
            self.current_scan = None
            
            # Update plot with original values
            self.update_plot()

    def continue_scan(self) -> None:
        """
        Continue the scanning process with the next value.
        This function:
        1. Checks if scanning should continue
        2. Updates coefficient values
        3. Updates the plot
        4. Schedules the next scan step

        继续扫描过程，使用下一个值。
        此函数：
        1. 检查是否应继续扫描
        2. 更新系数值
        3. 更新图形
        4. 安排下一个扫描步骤
        """
        if not self.scanning or not self.current_scan:
            return
            
        current_value = self.current_scan['value']
        order = self.current_scan['order']
        
        if current_value > SCAN_RANGE[1]:
            self.stop_scan()
            return
            
        # Update coefficients
        original_a = self.current_scan['original_a']
        original_b = self.current_scan['original_b']
        
        self.entries_a[order].delete(0, tk.END)
        self.entries_b[order].delete(0, tk.END)
        self.entries_a[order].insert(0, str(original_a * current_value))
        self.entries_b[order].insert(0, str(original_b * current_value))
        
        # Update plot
        self.update_plot()
        
        # Schedule next scan
        self.current_scan['value'] += SCAN_STEP
        self.root.after(SCAN_DELAY, self.continue_scan)

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
    
    app = MatrixPolynomialApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
