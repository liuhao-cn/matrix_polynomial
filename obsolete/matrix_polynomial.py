"""
Matrix Polynomial Visualization Tool

This module provides an interactive GUI application for visualizing 2D matrix transformations
and polynomial distortions. It allows users to explore how different grid patterns transform
under matrix operations and polynomial mappings.

The transformation is defined by:
    P(x,y) = sum_{i=0}^{n-1} (a_i I + b_i G)(x I + y G)^{i+1}
where:
    - I is the 2x2 identity matrix
    - G is a user-defined 2x2 matrix
    - a_i, b_i are polynomial coefficients
    - n is the polynomial order (default: 6)

Features:
    - Multiple grid types (HV, H, V, RC, R, C)
    - Adjustable number of grid lines
    - Optional 45° rotation
    - Real-time parameter adjustments
    - Configuration persistence
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import os

# Grid Generation Constants
GRID_RANGE = (-2, 2)  # Range for x and y coordinates
NUM_POINTS = 1000  # Number of points for line generation
MIN_CIRCLE_RADIUS = 0.4  # Minimum radius for circular grids
MAX_CIRCLE_RADIUS = 2.0  # Maximum radius for circular grids

# GUI Constants
FONT_SIZE = 21
FONT_FAMILY = 'Arial'
BUTTON_PADDING = (10, 0)
ENTRY_WIDTH = 6
MATRIX_SIZE = 2
NUM_COEFFICIENTS = 6
COEFFICIENTS_PER_FRAME = 2

# Plot Constants
PLOT_DPI = 300
PLOT_LINEWIDTH = 1.5
PLOT_FONTSIZE = 12
PLOT_MARGIN = 1.1  # Plot range margin (10% extra)

# Grid Type Constants
GRID_TYPE_HV = 'Hori/Vert'
GRID_TYPE_H = 'Hori'
GRID_TYPE_V = 'Vert'
GRID_TYPE_RC = 'Rad/Circ'
GRID_TYPE_R = 'Rad'
GRID_TYPE_C = 'Circ'

# Line Style Constants
LINE_STYLE_SOLID = '-'
LINE_STYLE_DOTTED = '--'  # Changed from ':' to '--' for dashed style

class MatrixPolynomialApp:
    """
    Main application class for the Matrix Polynomial visualization tool.
    
    This class handles the GUI setup, user interactions, and visualization logic
    for the matrix polynomial application. It provides an interactive interface
    for exploring various grid patterns and their transformations under matrix
    operations and polynomial mappings.
    
    The transformation process involves:
    1. Grid pattern generation (HV, H, V, RC, R, C types)
    2. Matrix and polynomial coefficient input
    3. Transformation calculation
    4. Real-time visualization update
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Polynomial")
        
        # Set config file path
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix_polynomial_config.json')
        
        # Load saved configuration or use defaults
        self.config = self.load_config()
        
        # Configure all styles
        style = ttk.Style()
        style.configure('Transform.TButton', font=(FONT_FAMILY, FONT_SIZE), padding=BUTTON_PADDING)
        style.configure('Transform.Selected.TButton', font=(FONT_FAMILY, FONT_SIZE, 'bold'), padding=BUTTON_PADDING)
        style.configure('Title.TLabel', font=(FONT_FAMILY, FONT_SIZE, 'bold'))
        style.configure('Header.TLabel', font=(FONT_FAMILY, FONT_SIZE, 'bold'))
        style.configure('Big.TLabel', font=(FONT_FAMILY, FONT_SIZE))
        style.configure('Big.TButton', font=(FONT_FAMILY, FONT_SIZE, 'bold'))
        style.configure('Big.TEntry', font=(FONT_FAMILY, FONT_SIZE, 'bold'))
        
        # Configure modern switch style
        style.configure('Switch.TCheckbutton',
                       font=(FONT_FAMILY, FONT_SIZE),
                       indicatorsize=25,
                       indicatormargin=5,
                       padding=5,
                       background='#ffffff',
                       foreground='#000000')

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create input frame
        input_frame = ttk.Frame(main_frame, padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create entry fields for coefficients
        self.entries_a = []
        self.entries_b = []
        
        # Load saved configuration
        self.config = self.load_config()
        
        # Create headers
        header_a = ttk.Label(input_frame, text="a", style='Header.TLabel')
        header_b = ttk.Label(input_frame, text="b", style='Header.TLabel')
        header_a.grid(row=0, column=0, padx=5)
        header_b.grid(row=0, column=2, padx=5)
        
        # Create three frames for the parameter columns
        frames = []
        for i in range(3):
            frame = ttk.Frame(input_frame, padding="10")
            frame.grid(row=0, column=i+1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
            frames.append(frame)
            
        # Create headers for each frame
        for frame in frames:
            ttk.Label(frame, text="I", style='Header.TLabel').grid(row=0, column=1, padx=5)
            ttk.Label(frame, text="G", style='Header.TLabel').grid(row=0, column=2, padx=5)
        
        # Create entry fields for each order
        for i in range(NUM_COEFFICIENTS):
            frame_idx = i // COEFFICIENTS_PER_FRAME  # Determine which frame to use
            row_in_frame = (i % COEFFICIENTS_PER_FRAME) * 2 + 1  # Alternate between rows 1 and 3
            
            ttk.Label(frames[frame_idx], text=f"Order {i+1}:", style='Big.TLabel').grid(row=row_in_frame, column=0, padx=5, pady=2)
            entry_a = ttk.Entry(frames[frame_idx], width=ENTRY_WIDTH, style='Big.TEntry', font=(FONT_FAMILY, FONT_SIZE))
            entry_a.insert(0, str(self.config['coefficients_a'][i]))
            entry_a.grid(row=row_in_frame, column=1, padx=5, pady=2)
            entry_a.bind('<Return>', lambda e: self.transform())
            
            entry_b = ttk.Entry(frames[frame_idx], width=ENTRY_WIDTH, style='Big.TEntry', font=(FONT_FAMILY, FONT_SIZE))
            entry_b.insert(0, str(self.config['coefficients_b'][i]))
            entry_b.grid(row=row_in_frame, column=2, padx=5, pady=2)
            entry_b.bind('<Return>', lambda e: self.transform())
            
            self.entries_a.append(entry_a)
            self.entries_b.append(entry_b)
        
        # Create matrix input at left
        matrix_frame = ttk.Frame(input_frame, padding="10")
        matrix_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20)
        
        ttk.Label(matrix_frame, text="G Matrix:", style='Header.TLabel').grid(row=0, column=0, columnspan=2, padx=5, pady=2)
        
        # Create 2x2 matrix inputs
        self.matrix_entries = []
        for i in range(MATRIX_SIZE):
            row_entries = []
            for j in range(MATRIX_SIZE):
                entry = ttk.Entry(matrix_frame, width=ENTRY_WIDTH, style='Big.TEntry', font=(FONT_FAMILY, FONT_SIZE))
                entry.insert(0, str(self.config['matrix'][i][j]))
                entry.grid(row=i+1, column=j, padx=5, pady=2)
                entry.bind('<Return>', lambda e: self.transform())
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
        
        # Create transform button frame
        button_frame = ttk.Frame(input_frame, padding="10")
        button_frame.grid(row=0, column=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=0)
        
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
        
        ttk.Label(num_grids_frame, text="Num of lines", style='Big.TLabel').pack(side=tk.LEFT, padx=(0, 2))
        self.num_grids_entry = ttk.Entry(num_grids_frame, width=3, style='Big.TEntry', font=(FONT_FAMILY, FONT_SIZE))
        self.num_grids_entry.pack(side=tk.LEFT)
        self.num_grids_entry.insert(0, str(self.config.get('num_grids', 4)))
        self.num_grids_entry.bind('<Return>', lambda e: self.update_num_grids())
        self.num_grids_entry.bind('<FocusOut>', lambda e: self.update_num_grids())
        
        # Add rotation toggle button
        self.rotation_enabled = tk.BooleanVar(value=self.config.get('rotation_enabled', True))
        rotation_btn = ttk.Checkbutton(button_frame, text="45° Rot", 
                                     variable=self.rotation_enabled,
                                     command=self.transform,
                                     style='Switch.TCheckbutton')
        rotation_btn.grid(row=3, column=3, padx=4, pady=0, sticky=(tk.W, tk.E))
        
        # Set initial selection from config
        self.current_transform_type = self.config['transform_type']
        self.update_button_styles()
        
        # Create matplotlib figure
        plt.rcParams.update({
            'font.size': PLOT_FONTSIZE,  # Restore plot font size to 24
            'lines.linewidth': PLOT_LINEWIDTH,
            'figure.dpi': PLOT_DPI,
            'savefig.dpi': PLOT_DPI
        })
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))  # Increase figure size for better text readability
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
            
    def matrix_power(self, x, y, power):
        """
        Compute (xI + yG)^power where G is the user-defined matrix.
        
        Args:
            x (float): Coefficient for the identity matrix
            y (float): Coefficient for the user-defined matrix
            power (int): Power to which the matrix should be raised
        
        Returns:
            np.ndarray: Resulting matrix
        """
        if power == 0:
            return np.eye(MATRIX_SIZE)
        
        # Get I and the current G matrix from UI
        I, G = self.get_matrix()

        # Base matrix xI + yG
        base = x * I + y * G
        
        # Use matrix power
        return np.linalg.matrix_power(base, power)
    
    def evaluate_polynomial(self, x, y, coeffs_a, coeffs_b):
        """
        Evaluate the polynomial at the given point (x, y).
        
        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            coeffs_a (list): Coefficients for the identity matrix terms
            coeffs_b (list): Coefficients for the user-defined matrix terms
        
        Returns:
            np.ndarray: Resulting matrix
        """
        result = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        I, G = self.get_matrix()
        
        # Pre-compute the base matrix and its powers
        base = x * I + y * G
        power_terms = [np.eye(MATRIX_SIZE)]  # Power 0
        for i in range(NUM_COEFFICIENTS):
            power_terms.append(power_terms[-1] @ base)
        
        # Compute the polynomial using pre-computed powers
        for i in range(NUM_COEFFICIENTS):
            coeff_matrix = coeffs_a[i] * I + coeffs_b[i] * G
            result += coeff_matrix @ power_terms[i+1]
            
        return result
    
    def on_closing(self):
        """
        Handle window closing event.
        """
        try:
            self.save_config()
            # Clean up matplotlib resources
            plt.close(self.fig)
            self.canvas.get_tk_widget().destroy()
        finally:
            self.root.quit()
            self.root.destroy()

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
        # Only radial lines
        points = []
        num_grids = self.get_num_grids()
        for i in range(num_grids):  # Use NUM_GRIDS for consistent coloring
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
        # Only circles
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
        Transform the given points using the current matrix and polynomial coefficients.
        
        Args:
            points (list): List of (x, y) points
        
        Returns:
            list: List of transformed (x, y) points
        """
        I, G = self.get_matrix()
        coeffs_a, coeffs_b = self.get_coefficients()
        G_T = G.transpose()
        
        transformed_points = []
        for x, y in points:
            # Vectorize the transformation
            x = np.array(x)
            y = np.array(y)
            
            # Pre-compute matrices for all points at once
            P_matrices = np.array([self.evaluate_polynomial(xi, yi, coeffs_a, coeffs_b) 
                                 for xi, yi in zip(x, y)])
            
            # Calculate traces using vectorized operations
            P_1 = np.trace(P_matrices, axis1=1, axis2=2) / 2
            P_2 = np.trace(G_T @ P_matrices.transpose(0, 2, 1), axis1=1, axis2=2) / 2
            
            # Apply final transformation
            if self.rotation_enabled.get():
                # Apply 45-degree rotation: x' = (P1-P2)/√2, y' = (P1+P2)/√2
                new_x = (P_1 - P_2) / np.sqrt(2)
                new_y = (P_1 + P_2) / np.sqrt(2)
            else:
                new_x, new_y = P_1, P_2
            
            transformed_points.append((new_x, new_y))
        return transformed_points
    
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
        max_range = 0
        
        # Get line style based on grid type
        def get_line_style(idx, title_suffix):
            if GRID_TYPE_HV in title_suffix:
                # For HV: horizontal (first half) solid, vertical (second half) dashed
                return LINE_STYLE_SOLID if idx < self.get_num_grids() else LINE_STYLE_DOTTED
            elif GRID_TYPE_RC in title_suffix:
                # For RC: circular (first half) solid, radial (second half) dashed
                return LINE_STYLE_SOLID if idx < self.get_num_grids() else LINE_STYLE_DOTTED
            elif GRID_TYPE_H in title_suffix or GRID_TYPE_C in title_suffix:
                # Single type H or C: all solid
                return LINE_STYLE_SOLID
            else:
                # Single type V or R: all dashed
                return LINE_STYLE_DOTTED
        
        # Get color based on grid type and index
        def get_color(idx, title_suffix):
            if GRID_TYPE_HV in title_suffix:
                # For HV: use same color for matching H/V pairs
                return f'C{(idx % self.get_num_grids())}'
            elif GRID_TYPE_RC in title_suffix:
                # For RC: use same color for matching R/C pairs
                return f'C{(idx % self.get_num_grids())}'
            elif GRID_TYPE_H in title_suffix or GRID_TYPE_V in title_suffix:
                # For H or V: use first set of colors
                return f'C{idx}'
            elif GRID_TYPE_C in title_suffix:
                # For C only: use first set of colors
                return f'C{idx}'
            else:
                # For R only: use first set of colors (same as C)
                return f'C{idx}'
        
        # Plot input points
        for idx, (x, y) in enumerate(input_points):
            style = get_line_style(idx, title_suffix)
            color = get_color(idx, title_suffix)
            self.ax1.plot(x, y, style, color=color, linewidth=PLOT_LINEWIDTH)
            max_range = max(max_range, np.max(np.abs([x, y])))
        
        # Plot transformed points
        transformed_points = self.transform_points(input_points)
        for idx, (x, y) in enumerate(transformed_points):
            style = get_line_style(idx, title_suffix)
            color = get_color(idx, title_suffix)
            self.ax2.plot(x, y, style, color=color, linewidth=PLOT_LINEWIDTH)
            max_range = max(max_range, np.max(np.abs([x, y])))
        
        # Set plot properties
        plot_range = max_range * PLOT_MARGIN
        for ax in [self.ax1, self.ax2]:
            ax.set_aspect('equal')
            ax.set_xlim(-plot_range, plot_range)
            ax.set_ylim(-plot_range, plot_range)
            ax.grid(True)
        
        self.ax1.set_title(f'$x$-$y$ plane', fontsize=PLOT_FONTSIZE)
        self.ax2.set_title(f'$u$-$v$ plane', fontsize=PLOT_FONTSIZE)
        
        # Update polynomial display
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
        self.poly_text = self.fig.text(0.5, 0.02, f"$u\\mathbf{{I}}+v\\mathbf{{G}}$ = {poly_str}", fontsize=PLOT_FONTSIZE-4, ha='center')
        
        # Adjust subplot spacing
        self.fig.subplots_adjust(bottom=0.15)
        
        # Update canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def transform_hv(self):
        """
        Transform the HV grid pattern.
        """
        self.plot_transformation(self.generate_points_hv(), f" ({GRID_TYPE_HV})")
    
    def transform_h(self):
        """
        Transform the H grid pattern.
        """
        self.plot_transformation(self.generate_points_h(), f" ({GRID_TYPE_H})")
    
    def transform_v(self):
        """
        Transform the V grid pattern.
        """
        self.plot_transformation(self.generate_points_v(), f" ({GRID_TYPE_V})")
    
    def transform_rc(self):
        """
        Transform the RC grid pattern.
        """
        self.plot_transformation(self.generate_points_rc(), f" ({GRID_TYPE_RC})")
    
    def transform_r(self):
        """
        Transform the R grid pattern.
        """
        self.plot_transformation(self.generate_points_r(), f" ({GRID_TYPE_R})")
    
    def transform_c(self):
        """
        Transform the C grid pattern.
        """
        self.plot_transformation(self.generate_points_c(), f" ({GRID_TYPE_C})")
    
    def transform(self):
        """
        Transform the current grid pattern.
        """
        # Call the appropriate transform function based on current type
        transform_functions = {
            GRID_TYPE_HV: self.transform_hv,
            GRID_TYPE_H: self.transform_h,
            GRID_TYPE_V: self.transform_v,
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
        """
        Update the styles of the transform buttons.
        """
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
            elif value > 12:
                value = 12
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

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')  # Maximize window
    app = MatrixPolynomialApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
