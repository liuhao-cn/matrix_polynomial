# Matrix Transform Visualization

A Python application for visualizing matrix transformations and polynomial distortions in 2D space. This tool helps in understanding how different types of grid patterns transform under matrix operations and polynomial mappings.

## Features

- Interactive visualization of 2D transformations
- Multiple grid pattern types (HV, H, V, RC, R, C)
- Adjustable number of grid lines
- Optional 45° rotation
- Real-time parameter adjustments
- Matrix and polynomial coefficient inputs
- Configurable visualization settings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/matrix-transform.git
cd matrix-transform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python matrix_transform.py
```

### Grid Types
- HV: Horizontal and Vertical lines
- H: Horizontal lines only
- V: Vertical lines only
- RC: Radial and Circular lines
- R: Radial lines only
- C: Circular lines only

### Controls
- Enter matrix coefficients in the G Matrix input fields
- Adjust polynomial coefficients using the a and b input fields
- Select grid type using the buttons
- Toggle 45° rotation with the switch
- Adjust number of grid lines using the input box

## Configuration

The application saves its configuration in `matrix_transform_config.json`. This includes:
- Last used matrix coefficients
- Polynomial coefficients
- Grid type selection
- Number of grid lines
- Rotation state

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
