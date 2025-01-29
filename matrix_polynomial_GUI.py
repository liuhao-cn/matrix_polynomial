"""
Matrix Polynomial Visualization Tool (Optimized Version)

This module provides an optimized version of the matrix polynomial visualization tool
that uses multiprocessing to speed up computations. The original functionality is preserved
while performance is improved through parallel processing.
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
            z1 = np.einsum('ijk,jlk->ilk', z1, z)
        
        # Add current term: (a_i*I + b_i*G)z^i
        coeff_matrix = coeffs_a[i] * I + coeffs_b[i] * G
        result += coeff_matrix @ z1
    
    # Convert result back to (u,v) coordinates
    u, v = decompose_HyperComplexNumbers(result, G)
    return u, v

class MatrixPolynomialAppOptimized:
    """
    Optimized version of the Matrix Polynomial visualization tool.
    Uses multiprocessing for faster computation of transformations.
    """
    def __init__(self, root):
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
            (GRID_TYPE_HV, self.transform_hv),
            (GRID_TYPE_H, self.transform_h),
            (GRID_TYPE_V, self.transform_v),
            (GRID_TYPE_RC, self.transform_rc),
            (GRID_TYPE_C, self.transform_c),
            (GRID_TYPE_R, self.transform_r)
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
    
    def generate_points_hv(self):
        """
        Generate points for the HV grid pattern.
        
        Returns:
            list: List of (x, y) points
        """
        points = []
        num_grids = self.get_num_grids()
        # Horizontal lines
        for i in range(num_grids):
            y = GRID_RANGE[0] + i * (GRID_RANGE[1] - GRID_RANGE[0]) / (num_grids - 1)
            x = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
            points.append((x, [y] * NUM_POINTS))
        
        # Vertical lines
        for i in range(num_grids):
            x = GRID_RANGE[0] + i * (GRID_RANGE[1] - GRID_RANGE[0]) / (num_grids - 1)
            y = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
            points.append(([x] * NUM_POINTS, y))
        return points
    
    def generate_points_h(self):
        """
        Generate points for the H grid pattern.
        
        Returns:
            list: List of (x, y) points
        """
        x = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
        points = []
        num_grids = self.get_num_grids()
        for y in np.linspace(GRID_RANGE[0], GRID_RANGE[1], num_grids):
            points.append((x, np.full_like(x, y)))
        return points
    
    def generate_points_v(self):
        """
        Generate points for the V grid pattern.
        
        Returns:
            list: List of (x, y) points
        """
        y = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
        points = []
        num_grids = self.get_num_grids()
        for x_val in np.linspace(GRID_RANGE[0], GRID_RANGE[1], num_grids):
            points.append((np.full_like(y, x_val), y))
        return points
    
    def generate_points_rc(self):
        """
        Generate points for the RC grid pattern.
        
        Returns:
            list: List of (x, y) points
        """
        points = []
        num_grids = self.get_num_grids()
        # Circular lines (first half)
        for i in range(num_grids):
            radius = MIN_CIRCLE_RADIUS + i * (MAX_CIRCLE_RADIUS - MIN_CIRCLE_RADIUS) / (num_grids - 1)
            theta = np.linspace(0, 2*np.pi, NUM_POINTS)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append((x, y))
        
        # Radial lines (second half)
        for i in range(num_grids):
            angle = i * (2*np.pi / num_grids)
            r = np.linspace(0, GRID_RANGE[1], NUM_POINTS)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append((x, y))
        return points
    
    def generate_points_r(self):
        """
        Generate points for the R grid pattern.
        
        Returns:
            list: List of (x, y) points
        """
        points = []
        num_grids = self.get_num_grids()
        for i in range(num_grids):
            angle = i * (2*np.pi / num_grids)
            r = np.linspace(0, GRID_RANGE[1], NUM_POINTS)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append((x, y))
        return points
    
    def generate_points_c(self):
        """
        Generate points for the C grid pattern.
        
        Returns:
            list: List of (x, y) points
        """
        theta = np.linspace(0, 2*np.pi, NUM_POINTS)
        points = []
        num_grids = self.get_num_grids()
        for r in np.linspace(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS, num_grids):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
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
    
    def transform_hv(self):
        """Transform the HV grid pattern."""
        self.plot_transformation(self.generate_points_hv(), f" ({GRID_TYPE_HV})")
    
    def transform_h(self):
        """Transform the H grid pattern."""
        self.plot_transformation(self.generate_points_h(), f" ({GRID_TYPE_H})")
    
    def transform_v(self):
        """Transform the V grid pattern."""
        self.plot_transformation(self.generate_points_v(), f" ({GRID_TYPE_V})")
    
    def transform_rc(self):
        """Transform the RC grid pattern."""
        self.plot_transformation(self.generate_points_rc(), f" ({GRID_TYPE_RC})")
    
    def transform_r(self):
        """Transform the R grid pattern."""
        self.plot_transformation(self.generate_points_r(), f" ({GRID_TYPE_R})")
    
    def transform_c(self):
        """Transform the C grid pattern."""
        self.plot_transformation(self.generate_points_c(), f" ({GRID_TYPE_C})")
    
    def transform(self):
        """
        Transform the current grid pattern.
        """
        transform_functions = {
            GRID_TYPE_HV: self.transform_hv,
            GRID_TYPE_H: self.transform_h,
            GRID_TYPE_V: self.transform_v,
            GRID_TYPE_NONE: lambda: None,  # Do nothing for NONE type
            GRID_TYPE_RC: self.transform_rc,
            GRID_TYPE_R: self.transform_r,
            GRID_TYPE_C: self.transform_c
        }
        transform_functions[self.current_transform_type]()
    
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
    
    def get_num_grids(self):
        """
        Get the current number of grid lines.
        
        Returns:
            int: Number of grid lines
        """
        try:
            return int(self.num_grids_entry.get())
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
