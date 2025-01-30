# Matrix Polynomial Visualization Tool

A visualization tool for matrix polynomials of the form P(x*I + y*G), where I is the identity matrix and G is a user-defined 2x2 matrix.

## Features

- Interactive GUI for matrix G and polynomial coefficients input
- Six grid pattern options: H+V, H, V, C+R, C, R
- Real-time visualization with input-output plane comparison
- Adjustable plot range and grid density
- Coefficient scanning functionality
- High-quality PDF export
- Auto-scaling plot ranges

## Installation

1. Clone this repository:
```bash
git clone https://github.com/liuhao-cn/matrix_polynomial.git
cd matrix_polynomial
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the GUI application:
```bash
python matrix_polynomial_GUI.py
```

For detailed usage instructions, see [Usage Guide](docs/usage_guide.md).

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- tkinter

See `requirements.txt` for detailed version requirements.

## Documentation

- [Usage Guide](docs/usage_guide.md): Detailed instructions for using the tool
- [Examples](docs/examples.md): Common use cases and example configurations

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
