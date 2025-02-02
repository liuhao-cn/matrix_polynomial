# Polynomial Distortion Correction System

A matrix polynomial-based image distortion correction solution

## Key Features

- Polynomial-based distortion model
- High-precision bilinear interpolation
- Multi-threaded acceleration
- Pure Python implementation

## Technical Advantages

- No compilation required
- Optimized matrix operations with NumPy
- Intuitive visualization tools

## Installation & Usage

### Prerequisites
- Python 3.7+
- NumPy 1.19+
- OpenCV 4.0+
- Matplotlib 3.3+

### Basic Usage
```python
import fix_distortion as fd

# Load and correct image
img = fd.load_image("distorted.png")
corrected = fd.correct_distortion(img)

# Save result
fd.save_image("corrected.png", corrected)
```

## Core Functionality

1. **Grid Generation**
   - Configurable grid density and size
   - Anti-aliasing with Gaussian blur

2. **Distortion Simulation**
   - Polynomial-based distortion model
   - Configurable Gaussian noise

3. **Parameter Estimation**
   - Random feature point sampling
   - Least squares coefficient estimation
   - Automatic ground truth comparison

4. **Image Correction**
   - Inverse polynomial transformation
   - High-precision interpolation

5. **Visualization**
   - Multi-window comparison
   - RGB channel overlay

## Quick Start

1. Clone repository:
```bash
git clone https://github.com/example/distortion_correction.git
cd distortion_correction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run demo:
```bash
python src/fix_distortion.py
```

## Implementation Details

### Coordinate System
- Normalized coordinates [-1, 1]
- Automatic coordinate mapping

### Mathematical Model
- Matrix polynomial transformation
- Support for high-order polynomials

### Image Processing
- OpenCV-based remapping
- Precision interpolation algorithms

## System Requirements

- Python 3.7+
- OpenCV 4.0+
- NumPy 1.19+
- Matplotlib 3.3+

## License

MIT License

---

# 多项式图像畸变校正系统

基于矩阵多项式的图像畸变校正解决方案

## 主要特性

- 基于多项式模型的畸变校正
- 高精度双线性插值算法
- 多线程加速计算
- 纯Python实现

## 技术优势

- 无需编译安装
- 基于NumPy的矩阵运算优化
- 直观的可视化工具

## 安装与使用

### 环境要求
- Python 3.7+
- NumPy 1.19+
- OpenCV 4.0+
- Matplotlib 3.3+

### 基本用法
```python
import fix_distortion as fd

# 加载并校正图像
img = fd.load_image("distorted.png")
corrected = fd.correct_distortion(img)

# 保存结果
fd.save_image("corrected.png", corrected)
```

## 核心功能

1. **网格生成**
   - 可配置网格密度和尺寸
   - 高斯模糊抗锯齿处理

2. **畸变模拟**
   - 基于多项式模型的畸变变换
   - 可配置高斯噪声

3. **参数估计**
   - 随机特征点采样
   - 最小二乘法系数估计
   - 自动真实值对比

4. **图像校正**
   - 逆多项式变换
   - 高精度插值算法

5. **可视化**
   - 多窗口对比显示
   - RGB通道叠加对比

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/example/distortion_correction.git
cd distortion_correction
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行演示：
```bash
python src/fix_distortion.py
```

## 实现细节

### 坐标系统
- 归一化坐标 [-1, 1]
- 自动坐标映射

### 数学模型
- 矩阵多项式变换
- 支持高阶多项式

### 图像处理
- 基于OpenCV的重映射
- 高精度插值算法

## 系统要求

- Python 3.7+
- OpenCV 4.0+
- NumPy 1.19+
- Matplotlib 3.3+

## 许可协议

MIT 许可证

## 作者信息

[作者信息]

## 版本历史

### v1.0.0 (2024-03-xx)
- 初始版本发布
- 实现基本的畸变校正功能
- 添加可视化对比功能