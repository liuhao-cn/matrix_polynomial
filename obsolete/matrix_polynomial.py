"""
Matrix Polynomial Batch Processing Tool

This script generates PDF plots of matrix polynomial transformations.
Modify the parameters below to customize the output.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which is more stable
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import platform
import os

# User Parameters
MATRIX = np.array([
    [0.0, 1.0],
    [1.0, 0.0]
])

COEFF_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Real coefficients
COEFF_B = [0.0, -0.2, 0.1, 0.05, 0.0, 0.0]  # Imaginary coefficients

GRID_TYPES = ['c']  # Options: 'hv', 'h', 'v', 'rc', 'r', 'c'
NUM_GRIDS = 4  # Number of grid lines
OUTPUT_DIR = 'output'  # Output directory for PDF files
ROTATION_ENABLED = True  # Whether to apply 45-degree rotation

# Constants
GRID_RANGE = (-2, 2)
NUM_POINTS = 5000
MIN_CIRCLE_RADIUS = 0.4
MAX_CIRCLE_RADIUS = 2.0
MATRIX_SIZE = 2
NUM_COEFFICIENTS = 6
PLOT_MARGIN = 1.1
NUM_PROCESSES = mp.cpu_count()

# Line Style Constants
LINE_STYLE_SOLID = '-'
LINE_STYLE_DOTTED = ':'

# Plot Settings
PLOT_DPI = 300
PLOT_LINEWIDTH = 1.5
PLOT_FONTSIZE = 12
FIGURE_SIZE = (12, 6)

def matrix_power(x, y, power, I, G):
    """Compute (xI + yG)^power where G is the user-defined matrix."""
    if power == 0:
        return np.eye(MATRIX_SIZE)
    base = x * I + y * G
    return np.linalg.matrix_power(base, power)

def evaluate_polynomial_chunk(chunk_data):
    """Evaluate polynomial for a chunk of points."""
    x_chunk, y_chunk, I, G, coeffs_a, coeffs_b = chunk_data
    P_matrices = np.array([evaluate_single_point(xi, yi, I, G, coeffs_a, coeffs_b) 
                          for xi, yi in zip(x_chunk, y_chunk)])
    G_T = G.transpose()
    P_1 = np.trace(P_matrices, axis1=1, axis2=2) / 2
    P_2 = np.trace(G_T @ P_matrices.transpose(0, 2, 1), axis1=1, axis2=2) / 2
    return P_1, P_2

def evaluate_single_point(x, y, I, G, coeffs_a, coeffs_b):
    """Evaluate polynomial at a single point."""
    # Pre-compute the base matrix and its powers
    base = x * I + y * G
    power_terms = [np.eye(MATRIX_SIZE)]  # Power 0
    for i in range(NUM_COEFFICIENTS):
        power_terms.append(power_terms[-1] @ base)
    
    # Compute the polynomial using pre-computed powers
    result = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for i in range(NUM_COEFFICIENTS):
        coeff_matrix = coeffs_a[i] * I + coeffs_b[i] * G
        result += coeff_matrix @ power_terms[i+1]
        
    return result

def transform_points(points, I, G, coeffs_a, coeffs_b, pool, rotation_enabled=True):
    """Transform points using parallel processing."""
    # Split points into chunks for parallel processing
    n = len(points)
    chunk_size = n // NUM_PROCESSES + (1 if n % NUM_PROCESSES else 0)
    chunks = [(points[i:i + chunk_size, 0], points[i:i + chunk_size, 1]) 
              for i in range(0, n, chunk_size)]
    
    # Prepare data for parallel processing
    chunk_data = [(x_chunk, y_chunk, I, G, coeffs_a, coeffs_b) 
                  for x_chunk, y_chunk in chunks]
    
    # Process chunks in parallel
    results = pool.map(evaluate_polynomial_chunk, chunk_data)
    
    # Combine results
    P_1 = np.concatenate([r[0] for r in results])
    P_2 = np.concatenate([r[1] for r in results])
    
    # Apply final transformation
    if rotation_enabled:
        # Apply 45-degree rotation: x' = (P1-P2)/√2, y' = (P1+P2)/√2
        new_x = (P_1 - P_2) / np.sqrt(2)
        new_y = (P_1 + P_2) / np.sqrt(2)
    else:
        new_x, new_y = P_1, P_2
    
    return new_x, new_y

def generate_grid_points(grid_type, num_grids):
    """Generate grid points based on the specified type."""
    points = []
    
    if grid_type in ['hv', 'h', 'v']:
        x = np.linspace(GRID_RANGE[0], GRID_RANGE[1], num_grids)
        y = np.linspace(GRID_RANGE[0], GRID_RANGE[1], num_grids)
        
        if grid_type in ['hv', 'h']:  # Horizontal lines
            for yi in y:
                t = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
                points.append(np.column_stack((t, np.full_like(t, yi))))
                
        if grid_type in ['hv', 'v']:  # Vertical lines
            for xi in x:
                t = np.linspace(GRID_RANGE[0], GRID_RANGE[1], NUM_POINTS)
                points.append(np.column_stack((np.full_like(t, xi), t)))
    
    elif grid_type in ['rc', 'r', 'c']:
        if grid_type in ['rc', 'r']:  # Radial lines
            for i in range(num_grids):
                angle = i * (2*np.pi / num_grids)
                r = np.linspace(0, GRID_RANGE[1], NUM_POINTS)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append(np.column_stack((x, y)))
                
        if grid_type in ['rc', 'c']:  # Circular lines
            radii = np.linspace(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS, num_grids)
            angles = np.linspace(0, 2*np.pi, NUM_POINTS)
            for radius in radii:
                x = radius * np.cos(angles)
                y = radius * np.sin(angles)
                points.append(np.column_stack((x, y)))
    
    return np.vstack(points)

def get_line_style(idx, grid_type, num_grids):
    """Get line style based on grid type and index."""
    if grid_type in ['hv', 'rc']:
        return LINE_STYLE_SOLID if idx < num_grids else LINE_STYLE_DOTTED
    elif grid_type in ['h', 'c']:
        return LINE_STYLE_SOLID
    else:
        return LINE_STYLE_DOTTED

def get_color(idx, grid_type, num_grids):
    """Get color based on grid type and index."""
    if grid_type in ['hv', 'rc']:
        return f'C{(idx % num_grids) % 10}'
    else:
        return f'C{idx % 10}'

def plot_transformation(input_points, transformed_points, grid_type, num_grids, output_file):
    """Plot the transformation and save to PDF."""
    # Set up the figure with proper styling
    plt.rcParams.update({
        'font.size': PLOT_FONTSIZE,
        'lines.linewidth': PLOT_LINEWIDTH,
        'figure.dpi': PLOT_DPI,
        'savefig.dpi': PLOT_DPI
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # Track maximum range for plot limits
    max_range = 0
    
    # Calculate number of lines
    num_lines = len(input_points) // NUM_POINTS
    
    # Plot original points
    for idx in range(num_lines):
        start_idx = idx * NUM_POINTS
        end_idx = (idx + 1) * NUM_POINTS
        x = input_points[start_idx:end_idx, 0]
        y = input_points[start_idx:end_idx, 1]
        style = get_line_style(idx, grid_type, num_grids)
        color = get_color(idx, grid_type, num_grids)
        ax1.plot(x, y, style, color=color, linewidth=PLOT_LINEWIDTH)
        max_range = max(max_range, np.max(np.abs([x, y])))
    
    # Plot transformed points
    transformed_x, transformed_y = transformed_points
    for idx in range(num_lines):
        start_idx = idx * NUM_POINTS
        end_idx = (idx + 1) * NUM_POINTS
        x = transformed_x[start_idx:end_idx]
        y = transformed_y[start_idx:end_idx]
        style = get_line_style(idx, grid_type, num_grids)
        color = get_color(idx, grid_type, num_grids)
        ax2.plot(x, y, style, color=color, linewidth=PLOT_LINEWIDTH)
        max_range = max(max_range, np.max(np.abs([x, y])))
    
    # Set plot properties
    plot_range = max_range * PLOT_MARGIN
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.grid(True)
    
    # Set titles with grid type
    grid_type_map = {
        'hv': 'Horizontal/Vertical',
        'h': 'Horizontal',
        'v': 'Vertical',
        'rc': 'Radial/Circular',
        'r': 'Radial',
        'c': 'Circular'
    }
    grid_type_title = grid_type_map.get(grid_type, grid_type)
    ax1.set_title(f'$x$-$y$ plane ({grid_type_title})', fontsize=PLOT_FONTSIZE)
    ax2.set_title(f'$u$-$v$ plane ({grid_type_title})', fontsize=PLOT_FONTSIZE)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate plots."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize multiprocessing pool
    pool = mp.Pool(NUM_PROCESSES)
    
    # Create identity matrix
    I = np.eye(MATRIX_SIZE)
    
    # Process each grid type
    for grid_type in GRID_TYPES:
        # Generate grid points
        input_points = generate_grid_points(grid_type.lower(), NUM_GRIDS)
        
        # Transform points
        transformed_points = transform_points(input_points, I, MATRIX, COEFF_A, COEFF_B, pool, ROTATION_ENABLED)
        
        # Plot and save
        output_file = f"{OUTPUT_DIR}/matrix_polynomial.pdf"
        plot_transformation(input_points, transformed_points, grid_type.lower(), NUM_GRIDS, output_file)
    
    # Clean up
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Enable multiprocessing for Windows
    mp.freeze_support()
    main()
