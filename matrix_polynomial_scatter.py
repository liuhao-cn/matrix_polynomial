"""
Matrix Polynomial Scatter Visualization

This module provides a GUI application for visualizing matrix polynomial transformations
using scatter plots and density-based visualization. It offers real-time visualization
of how points in 2D space are transformed under matrix polynomial operations.

Key Features:
- High-performance scatter plot visualization
- Density-based color mapping
- Support for millions of points
- Position-based and polar coordinate-based coloring
- Configurable binning resolution
- Real-time parameter adjustments

The visualization uses efficient binning and averaging techniques to handle
large point sets while maintaining interactive performance.

Dependencies:
    numpy (>=1.24.0): For numerical computations
    matplotlib (>=3.7.0): For plotting
    tkinter: For GUI components
    json: For configuration handling
    
Author: Hao Liu
License: MIT
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import os
import platform
import time

# Grid Generation Constants
GRID_RANGE = (-2, 2)  # Range for x and y coordinates
NUM_POINTS = 1500000    # Number of points for visualization
SCATTER_POINT_SIZE = 4  # Size of scatter points
BINS_PER_AXIS = 400    # Resolution of binning grid

# GUI Constants for Windows
WIN_FONT_SIZE = 23
WIN_BUTTON_PADDING = (10, 0)
WIN_ENTRY_WIDTH = 7
WIN_PLOT_DPI = 150
WIN_PLOT_LINEWIDTH = 0.7
WIN_PLOT_FONTSIZE = 20
WIN_FIGURE_SIZE = (16, 8)

# GUI Constants for Linux
LINUX_FONT_SIZE = 11
LINUX_BUTTON_PADDING = (5, 0)
LINUX_ENTRY_WIDTH = 5
LINUX_PLOT_DPI = 150
LINUX_PLOT_LINEWIDTH = 1.0
LINUX_PLOT_FONTSIZE = 8
LINUX_FIGURE_SIZE = (8, 4)

# Common parameters
FONT_FAMILY = 'Arial'
MATRIX_SIZE = 2  # Size of the transformation matrix
NUM_COEFFICIENTS = 6  # Number of polynomial coefficients
COEFFICIENTS_PER_FRAME = 2  # Number of coefficients shown per frame
PLOT_MARGIN = 1.1  # Plot range margin (10% extra)

# Color scheme options
COLOR_SCHEME_XY = 'xy'  # Color based on x-y coordinates
COLOR_SCHEME_RADIAL = 'radial'  # Color based on radius and angle

class MatrixPolynomialAppOptimized:
    """
    A GUI application for visualizing matrix polynomial transformations using scatter plots.
    
    This class implements an optimized visualization approach using density-based rendering
    and efficient binning techniques. It can handle millions of points while maintaining
    interactive performance.
    
    Features:
        - High-performance point generation and transformation
        - Density-based visualization with configurable binning
        - Position-based and polar coordinate-based coloring
        - Real-time parameter adjustments
        - Performance monitoring with timing breakdowns
    
    The visualization process includes:
        1. Point generation in input space
        2. Matrix polynomial transformation
        3. Color computation based on point positions
        4. Efficient binning and density computation
        5. Interactive display with real-time updates
    
    All GUI elements are scaled appropriately for both Windows and Linux platforms.
    Performance is optimized through vectorized operations and efficient binning algorithms.
    """

    # Number of blocks for binning (resolution of the visualization)

    @staticmethod
    def mm22_series(z1, z2):
        """
        Multiply two 2x2 matrices element-wise for a series of points.
        
        Args:
            z1 (ndarray): First 2x2xN array of matrices
            z2 (ndarray): Second 2x2xN array of matrices
        
        Returns:
            ndarray: Result of matrix multiplication with shape 2x2xN
        """
        z = z1*0  # Initialize result array with same shape as input
        z[0,0] = z1[0,0]*z2[0,0] + z1[0,1]*z2[1,0]
        z[0,1] = z1[0,0]*z2[0,1] + z1[0,1]*z2[1,1]
        z[1,0] = z1[1,0]*z2[0,0] + z1[1,1]*z2[1,0]
        z[1,1] = z1[1,0]*z2[0,1] + z1[1,1]*z2[1,1]
        return z

    @staticmethod
    def decompose_HyperComplexNumbers(z, G):
        """
        Decompose a series of hypercomplex numbers into their components.
        
        Args:
            z (ndarray): Array of 2x2 matrices representing hypercomplex numbers
            G (ndarray): The basis matrix G
        
        Returns:
            tuple: (x, y) components in the I-G basis
        """
        x = np.trace(z)/2  # Extract x component using matrix trace
        y = np.trace(G.transpose() @ z)/2  # Extract y component using G-projection
        return x, y

    @staticmethod
    def compute_polynomial(x, y, I, G, coeffs_a, coeffs_b):
        """
        Evaluate the matrix polynomial P(z) = sum(a_i*I + b_i*G)(z)^i.
        
        Args:
            x (ndarray): x-coordinates of input points
            y (ndarray): y-coordinates of input points
            I (ndarray): 2x2 identity matrix
            G (ndarray): 2x2 basis matrix
            coeffs_a (list): Coefficients for I terms
            coeffs_b (list): Coefficients for G terms
        
        Returns:
            tuple: (u, v) coordinates of transformed points
        """
        # Initialize result matrix
        result = np.zeros((MATRIX_SIZE, MATRIX_SIZE, NUM_POINTS))
        
        # Compute powers of z = xI + yG and accumulate terms
        for i in range(NUM_COEFFICIENTS):
            if i==0:
                # First iteration: z = xI + yG
                z = np.outer(np.ravel(I), x) + np.outer(np.ravel(G), y)
                z = np.reshape(z, (2, 2, NUM_POINTS))
                z1 = z.copy()
            else:
                # Higher powers: z1 = z1 * z
                z1 = MatrixPolynomialAppOptimized.mm22_series(z1, z)
            
            # Add current term: (a_i*I + b_i*G)z^i
            coeff_matrix = coeffs_a[i] * I + coeffs_b[i] * G
            result += coeff_matrix @ z1
        
        # Convert result back to (u,v) coordinates
        u, v = MatrixPolynomialAppOptimized.decompose_HyperComplexNumbers(result, G)
        return u, v

    @staticmethod
    def compute_binned_colors(x, y, colors, x_range, y_range, nbins):
        """
        Compute average colors for bins containing points using fast numpy operations.
        
        Args:
            x, y (ndarray): Point coordinates
            colors (ndarray): RGB colors for each point
            x_range, y_range (tuple): Range for x and y axes
            nbins (int): Number of bins per axis
            
        Returns:
            tuple: Arrays of bin edges (x, y) and colors
        """
        t0 = time.time()
        
        x_edges = np.linspace(x_range[0], x_range[1], nbins + 1)
        y_edges = np.linspace(y_range[0], y_range[1], nbins + 1)
        
        # Compute histograms for each color channel
        weights_r = colors[:, 0]
        weights_g = colors[:, 1]
        weights_b = colors[:, 2]
        
        t1 = time.time()
        # Compute sum of colors and counts in one pass
        H_r, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=weights_r)
        H_g, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=weights_g)
        H_b, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=weights_b)
        H_counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
        
        t2 = time.time()
        # Stack the color channels
        bin_colors = np.stack([H_r, H_g, H_b], axis=-1)
        
        # Compute averages, avoiding division by zero
        mask = H_counts > 0
        bin_colors[mask] /= H_counts[mask, np.newaxis]
        
        t3 = time.time()
        print(f"Binning setup: {(t1-t0)*1000:.1f}ms")
        print(f"Histogram computation: {(t2-t1)*1000:.1f}ms")
        print(f"Color processing: {(t3-t2)*1000:.1f}ms")
        print(f"Total binning time: {(t3-t0)*1000:.1f}ms")
        
        return x_edges, y_edges, bin_colors

    def __init__(self, root):
        """
        Initialize the application with the given root window.

        Args:
            root: The Tkinter root window
        """
        self.root = root
        self.root.title("Matrix Polynomial Scatter")
        
        # Set config file path
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix_polynomial_scatter_config.json')
        
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
        
        # Create color scheme toggle frame
        button_frame = ttk.Frame(input_frame, padding="10")
        button_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=0)
        
        # Add color scheme radio buttons
        ttk.Label(button_frame, text="Color Scheme", style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        self.color_scheme = tk.StringVar(value=self.config.get('color_scheme', COLOR_SCHEME_XY))
        
        ttk.Radiobutton(button_frame, text="X-Y Colors", variable=self.color_scheme, 
                       value=COLOR_SCHEME_XY, command=self.transform,
                       style='Switch.TCheckbutton').grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Radiobutton(button_frame, text="Radial Colors", variable=self.color_scheme,
                       value=COLOR_SCHEME_RADIAL, command=self.transform,
                       style='Switch.TCheckbutton').grid(row=1, column=1, padx=5, pady=5)
        
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
    
    def generate_scatter_points(self):
        """
        Generate random points within the grid range for visualization.

        Returns:
            tuple: (x, y) coordinates of generated points
        """
        points_x = np.random.rand(NUM_POINTS)
        points_y = np.random.rand(NUM_POINTS)

        # Scale points to the grid range
        points_x = points_x*(GRID_RANGE[1]-GRID_RANGE[0]) + GRID_RANGE[0]
        points_y = points_y*(GRID_RANGE[1]-GRID_RANGE[0]) + GRID_RANGE[0]
            
        return points_x, points_y
        
    def get_colors(self, x, y):
        """
        Compute colors for points based on their position and current color scheme.

        Args:
            x (ndarray): x-coordinates of points
            y (ndarray): y-coordinates of points

        Returns:
            ndarray: RGB colors for each point, shape (N, 3)
        """
        if self.color_scheme.get() == COLOR_SCHEME_XY:
            # Color based on x,y position
            colors = np.zeros((len(x), 3))
            # Normalize x and y to [0,1] range for RGB
            colors[:, 0] = (x + 2) / 4  # R channel from x
            colors[:, 1] = (y + 2) / 4  # G channel from y
            colors[:, 2] = 1 - ((x + 2) / 4 + (y + 2) / 4) / 2  # B inverse channel from average
        else:
            # Color based on radius and angle
            r = np.sqrt(x**2 + y**2)  # Radius from origin
            theta = np.arctan2(y, x)   # Angle from x-axis
            
            colors = np.zeros((len(x), 3))
            colors[:, 0] = r / np.sqrt(8)  # R channel from radius (max radius is sqrt(8))
            colors[:, 1] = (theta + np.pi) / (2 * np.pi)  # G channel from angle
            colors[:, 2] = 1 - r / np.sqrt(8)  # B channel inverse of radius
        
        return colors
        
    def plot_transformation(self, title_suffix=""):
        """
        Plot the input points and their transformed positions.

        This method:
        1. Generates random points in the input plane
        2. Applies the matrix polynomial transformation
        3. Colors the points based on the selected scheme
        4. Displays both input and output planes
        5. Shows the polynomial equation

        Args:
            title_suffix (str): Optional suffix for plot titles
        """
        t_start = time.time()
        
        # Clear the plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Generate scatter points
        t0 = time.time()
        x, y = self.generate_scatter_points()
        t1 = time.time()
        
        # Get matrix and coefficients
        I, G = self.get_matrix()
        coeffs_a, coeffs_b = self.get_coefficients()
        
        # Process points in parallel
        t2 = time.time()
        u, v = self.compute_polynomial(x, y, I, G, coeffs_a, coeffs_b)
        t3 = time.time()
        
        # Get colors based on original points
        colors = self.get_colors(x, y)
        t4 = time.time()
        
        # Compute binned colors for both plots
        print("\nProcessing input plane:")
        x_edges1, y_edges1, colors1 = self.compute_binned_colors(x, y, colors, 
                                                           GRID_RANGE, GRID_RANGE, 
                                                           BINS_PER_AXIS)
        
        print("\nProcessing output plane:")
        max_range = max(np.max(np.abs(u)), np.max(np.abs(v)))
        plot_range = max_range * 1.1
        x_edges2, y_edges2, colors2 = self.compute_binned_colors(u, v, colors, 
                                                           (-plot_range, plot_range),
                                                           (-plot_range, plot_range),
                                                           BINS_PER_AXIS)
        t5 = time.time()
        
        # Plot using pcolormesh for efficiency
        self.ax1.pcolormesh(x_edges1, y_edges1, colors1, shading='flat')
        self.ax2.pcolormesh(x_edges2, y_edges2, colors2, shading='flat')
        
        # Set properties for input plot (ax1)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlim(-2, 2)
        self.ax1.set_ylim(-2, 2)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title(f'$x$-$y$ plane', fontsize=self.current_plot_fontsize)
        
        # Set properties for transformed plot (ax2)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(-plot_range, plot_range)
        self.ax2.set_ylim(-plot_range, plot_range)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title(f'$u$-$v$ plane', fontsize=self.current_plot_fontsize)
        
        t6 = time.time()
        
        print(f"\nTiming breakdown:")
        print(f"Point generation: {(t1-t0)*1000:.1f}ms")
        print(f"Matrix setup: {(t2-t1)*1000:.1f}ms")
        print(f"Polynomial computation: {(t3-t2)*1000:.1f}ms")
        print(f"Color computation: {(t4-t3)*1000:.1f}ms")
        print(f"Binning and averaging: {(t5-t4)*1000:.1f}ms")
        print(f"Plot rendering: {(t6-t5)*1000:.1f}ms")
        print(f"Total time: {(t6-t_start)*1000:.1f}ms")
        
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
        self.fig.subplots_adjust(bottom=0.17, wspace=0.2, top=0.92, left=0.08, right=0.98)
        
        # Update canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def transform(self):
        """Transform the current points."""
        self.plot_transformation()
        self.save_config()
        
    def save_config(self):
        """Save current configuration to JSON file."""
        config = {
            'matrix': [[float(entry.get()) for entry in row] for row in self.matrix_entries],
            'coefficients_a': [float(entry.get()) for entry in self.entries_a],
            'coefficients_b': [float(entry.get()) for entry in self.entries_b],
            'color_scheme': self.color_scheme.get()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
    
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
            'color_scheme': COLOR_SCHEME_XY
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
    
    def get_matrix(self):
        """
        Get the current matrix from the input fields.
        
        Returns:
            tuple: (I, G) where I is the identity matrix and G is the user-defined matrix
        """
        I = np.eye(MATRIX_SIZE)
        G = np.array([[float(entry.get()) for entry in row] for row in self.matrix_entries])
        G = G/np.sqrt(np.abs(np.linalg.det(G)))
        return I, G
    
    def get_coefficients(self):
        """
        Get the current coefficients from the input fields.
        
        Returns:
            tuple: (coeffs_a, coeffs_b) where coeffs_a and coeffs_b are lists of coefficients
        """
        coeffs_a = [float(entry.get()) for entry in self.entries_a]
        coeffs_b = [float(entry.get()) for entry in self.entries_b]
        return coeffs_a, coeffs_b
    
    def on_closing(self):
        """
        Clean up resources when closing the application.
        """
        try:
            # Clean up matplotlib resources
            plt.close(self.fig)
            self.canvas.get_tk_widget().destroy()
        finally:
            self.root.quit()
            self.root.destroy()
    
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
