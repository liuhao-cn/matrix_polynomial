# Matrix Polynomial Visualization

A tool for visualizing matrix polynomial transformations in the complex or hypercomplex plane.

## Features

- Interactive matrix and coefficient input
- Multiple grid pattern options
- Real-time visualization
- PDF export functionality
- Coefficient scanning animation
- **Image Distortion Correction**: A practical application demonstrating matrix polynomial transformations

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

Run the distortion correction demo:
```bash
python fix_distortion.py
```

## Project Structure
- `matrix_polynomial_GUI.py`: Main GUI application
- `matrix_polynomial_math.py`: Core mathematical functions
- `matrix_polynomial_config.json`: Saved configuration
- `fix_distortion.py`: Demonstration of matrix polynomial application in image distortion correction

## Applications

### Image Distortion Correction
The `fix_distortion.py` demonstrates how matrix polynomials can be used to:
- Model complex image distortions using polynomial transformations
- Estimate distortion parameters from grid patterns
- Correct distorted images through inverse polynomial mapping

Example usage:
```python
from fix_distortion import apply_polynomial_distortion, apply_correction

# Generate test data
original_image = generate_original_grid()
distorted_image, *_ = apply_polynomial_distortion(original_image)

# Estimate and correct distortion
corrected_image = apply_correction(distorted_image, a_coeffs, b_coeffs)
```

## License
MIT License

---

# 矩阵多项式可视化

一个用于在复平面或超复平面上可视化矩阵多项式变换的工具。

## 功能特点

- 交互式矩阵和系数输入
- 多种网格模式选项
- 实时可视化
- PDF 导出功能
- 系数扫描动画
- **图像畸变校正**：矩阵多项式变换的实际应用示例

## 安装

1. 克隆此仓库：
```bash
git clone https://github.com/liuhao-cn/matrix_polynomial.git
cd matrix_polynomial
```

2. 安装所需包：
```bash
pip install -r requirements.txt
```

## 使用方法

运行 GUI 应用程序：
```bash
python matrix_polynomial_GUI.py
```

运行畸变校正演示：
```bash
python fix_distortion.py
```

## 项目结构
- `matrix_polynomial_GUI.py`：主 GUI 应用程序
- `matrix_polynomial_math.py`：核心数学函数
- `matrix_polynomial_config.json`：保存的配置
- `fix_distortion.py`：矩阵多项式在图像畸变校正中的应用演示

## 应用案例

### 图像畸变校正
`fix_distortion.py` 展示了矩阵多项式在以下方面的应用：
- 使用多项式变换对复杂图像畸变建模
- 从网格图案中估计畸变参数
- 通过逆多项式映射校正畸变图像

使用示例：
```python
from fix_distortion import apply_polynomial_distortion, apply_correction

# 生成测试数据
original_image = generate_original_grid()
distorted_image, *_ = apply_polynomial_distortion(original_image)

# 估计并校正畸变
corrected_image = apply_correction(distorted_image, a_coeffs, b_coeffs)
```

## 许可证
MIT 许可证