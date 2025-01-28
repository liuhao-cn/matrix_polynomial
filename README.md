# Matrix Polynomial Visualization

A Python application for visualizing matrix transformations and polynomial distortions in 2D space. This tool helps in understanding how different types of grid patterns and point distributions transform under matrix operations and polynomial mappings.

## Features

### Core Features
- Interactive GUI for real-time visualization of 2D transformations
- Multiple visualization modes:
  - Grid patterns (HV, H, V, RC, R, C)
  - Point scatter plots with density visualization
  - Color-mapped transformations
- Matrix and polynomial coefficient inputs
- Real-time parameter adjustments

### Visualization Options
- Adjustable grid density and point count
- Configurable plot resolution and binning
- Color mapping based on position or polar coordinates
- Performance optimized rendering for large datasets
- Support for high-resolution displays

## Installation

1. Clone the repository:
```bash
git clone https://github.com/liuhao-cn/matrix_polynomial.git
cd matrix_polynomial
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Grid Visualization
Run the grid-based visualization:
```bash
python matrix_polynomial_GUI.py
```

### Point Distribution Visualization
Run the scatter plot visualization:
```bash
python matrix_polynomial_scatter.py
```

### Command Line Interface
For basic transformations without GUI:
```bash
python matrix_polynomial.py
```

## Visualization Modes

### Grid Types
- HV: Horizontal and Vertical lines
- H: Horizontal lines only
- V: Vertical lines only
- RC: Radial and Circular lines
- R: Radial lines only
- C: Circular lines only

### Scatter Plot Features
- Position-based color mapping
- Density-aware visualization
- Adjustable point count (up to millions of points)
- Configurable resolution and binning
- Performance-optimized rendering

## Configuration

### GUI Settings
- Adjustable plot size and DPI
- Configurable font sizes
- Grid density controls
- Color scheme selection
- Real-time parameter updates

### Performance Settings
- Adjustable point count for scatter plots
- Configurable binning resolution
- Optimized rendering for large datasets

## File Structure
- `matrix_polynomial_GUI.py`: Main GUI application for grid visualization
- `matrix_polynomial_scatter.py`: Scatter plot visualization with density mapping
- `matrix_polynomial.py`: Core transformation logic and CLI interface
- `matrix_polynomial_config.json`: Configuration for grid visualization
- `matrix_polynomial_scatter_config.json`: Configuration for scatter plot visualization

## Requirements
- Python 3.8+
- NumPy 1.24.0+
- Matplotlib 3.7.0+
- Tkinter
- Pillow 10.0.0+
- SciPy 1.11.0+

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
