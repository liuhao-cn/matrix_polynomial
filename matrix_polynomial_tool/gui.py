import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import platform
import numpy as np
from math_utils import compute_polynomial, decompose_HyperComplexNumbers
from config_manager import load_config, save_config

class MatrixPolynomialAppOptimized:
    """
    Matrix Polynomial Visualization Tool GUI.
    
    This class provides a graphical interface for visualizing matrix polynomial transformations
    with various grid patterns and interactive controls.
    """
    
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
        """
        points = []
        num_grids = self.get_num_grids()
        
        # 生成直线网格点
        if 'Horizontal' in grid_type:
            # 水平线，x范围[-2,2]，y为固定值
            for i in range(num_grids):
                y = -2 + i * 4 / (num_grids - 1)  # y在[-2,2]范围内均匀分布
                x = np.linspace(-2, 2, 10000)     # shape: (10000,)
                points.append((x, np.full(10000, y)))  # y扩展为与x相同大小
        
        if 'Vertical' in grid_type:
            # 垂直线，y范围[-2,2]，x为固定值
            for i in range(num_grids):
                x = -2 + i * 4 / (num_grids - 1)  # x在[-2,2]范围内均匀分布
                y = np.linspace(-2, 2, 10000)     # shape: (10000,)
                points.append((np.full(10000, x), y))  # x扩展为与y相同大小
        
        # 生成圆形和径向网格点
        if 'Circular' in grid_type:
            # 圆形线，半径从0.1到2.0
            theta = np.linspace(0, 2*np.pi, 10000)  # shape: (10000,)
            for i in range(num_grids):
                r = 0.1 + i * 1.9 / (num_grids - 1)  # 半径在[0.1,2.0]范围内均匀分布
                x = r * np.cos(theta)  # shape: (10000,)
                y = r * np.sin(theta)  # shape: (10000,)
                points.append((x, y))
        
        if 'Radial' in grid_type:
            # 径向线，从原点发出
            r = np.linspace(0, 2, 10000)  # shape: (10000,)
            for i in range(num_grids):
                angle = i * 2*np.pi / num_grids  # 角度均匀分布在[0,2π]
                x = r * np.cos(angle)  # shape: (10000,)
                y = r * np.sin(angle)  # shape: (10000,)
                points.append((x, y))
        
        return points

    def transform_hv(self) -> None:
        """
        Transform using both horizontal and vertical lines.
        Generates a grid pattern with both horizontal and vertical lines
        and applies the current transformation.
        """
        points = self.generate_points('Horizontal+Vertical')
        self.plot_transformation(points, " (Horizontal+Vertical)")

    def transform_h(self) -> None:
        """
        Transform using horizontal lines only.
        Generates a pattern with only horizontal lines and applies
        the current transformation.
        """
        points = self.generate_points('Horizontal')
        self.plot_transformation(points, " (Horizontal)")

    def transform_v(self) -> None:
        """
        Transform using vertical lines only.
        Generates a pattern with only vertical lines and applies
        the current transformation.
        """
        points = self.generate_points('Vertical')
        self.plot_transformation(points, " (Vertical)")

    def transform_rc(self) -> None:
        """
        Transform using both radial and circular lines.
        Generates a pattern combining circular and radial lines
        and applies the current transformation.
        """
        points = self.generate_points('Circular+Radial')
        self.plot_transformation(points, " (Circular+Radial)")

    def transform_r(self) -> None:
        """
        Transform using radial lines only.
        Generates a pattern with only radial lines emanating from
        the origin and applies the current transformation.
        """
        points = self.generate_points('Radial')
        self.plot_transformation(points, " (Radial)")

    def transform_c(self) -> None:
        """
        Transform using circular lines only.
        Generates a pattern with only concentric circles and applies
        the current transformation.
        """
        points = self.generate_points('Circular')
        self.plot_transformation(points, " (Circular)")

    def plot_transformation(self, points: list[tuple[np.ndarray, np.ndarray]], 
                          title_suffix: str = "") -> None:
        """
        Plot the transformation of input points.
        
        Args:
            points (list[tuple[np.ndarray, np.ndarray]]): List of point pairs to transform,
                where each pair contains x and y coordinate arrays of shape (10000,)
            title_suffix (str, optional): Additional text to append to plot titles.
                Defaults to empty string.
        """
        self.ax1.clear()
        self.ax2.clear()
        
        # Transform points using current matrix and coefficients
        transformed_points = []
        I, G = self.get_matrix()
        coeffs_a, coeffs_b = self.get_coefficients()
        
        for x, y in points:
            u, v = compute_polynomial(x, y, I, G, coeffs_a, coeffs_b)
            if self.rotation_enabled.get():
                # Apply 45-degree rotation
                u1 = (u - v) / np.sqrt(2)
                v1 = (u + v) / np.sqrt(2)
            else:
                u1, v1 = u, v
            transformed_points.append((u1, v1))
        
        # Plot original and transformed points
        for i, ((x, y), (u, v)) in enumerate(zip(points, transformed_points)):
            line_style = '-' if i < self.get_num_grids() else '--'
            self.ax1.plot(x, y, line_style)
            self.ax2.plot(u, v, line_style)
        
        # Configure plot properties
        for ax in [self.ax1, self.ax2]:
            ax.set_aspect('equal')
            ax.grid(True)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
        
        self.ax1.set_title(f'Input plane{title_suffix}')
        self.ax2.set_title(f'Output plane{title_suffix}')
        
        self.canvas.draw()

# ... 其他方法的实现保持不变 ... 