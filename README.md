# 畸变校正系统 (Distortion Correction System)

一个基于矩阵多项式的图像畸变校正系统，通过网格图像演示了畸变建模、参数估计和校正过程。

## 功能特点

1. **网格图像生成**
   - 可配置网格密度和尺寸
   - 支持高斯模糊抗锯齿处理

2. **畸变模拟**
   - 基于多项式模型的畸变变换
   - 支持添加高斯噪声模拟真实情况

3. **参数估计**
   - 随机采样特征点对
   - 最小二乘法估计畸变系数
   - 自动对比真实与估计系数

4. **图像校正**
   - 基于逆多项式变换的校正
   - 高精度双线性插值重建

5. **结果可视化**
   - 多窗口对比显示
   - 支持红绿通道叠加对比

## 安装

1. 克隆此仓库：
```bash
git clone https://github.com/liuhao-cn/distortion_correction.git
cd distortion_correction
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

运行演示程序：
```bash
python src/fix_distortion.py
```

## 参数配置

在 src/fix_distortion.py 中可以调整以下参数：

```python
# 网格参数
GRID_COLS = 32           # 网格横向单元格数量
GRID_ROWS = 24           # 网格纵向单元格数量
CELL_SIZE = 50           # 单元格边长（像素）

# 畸变参数
POLY_COEFFS_A = [...]    # 多项式A系数
POLY_COEFFS_B = [...]    # 多项式B系数
NOISE_STD = 2.0          # 噪声标准差

# 校正参数
INTERP_RANGE = 2         # 插值邻域范围
SAMPLE_POINTS = 50       # 采样点数量
```

## 技术实现

1. **坐标系统**
   - 使用归一化坐标系统 [-1,1]
   - 自动处理坐标映射和逆映射

2. **数学模型**
   - 基于矩阵多项式的变换模型
   - 支持高阶多项式拟合

3. **图像处理**
   - OpenCV 实现图像重映射
   - 高精度插值算法

## 项目结构

```
distortion_correction/
├── README.md           # 项目说明文档
├── requirements.txt    # 依赖包列表
├── src/
│   ├── __init__.py
│   ├── fix_distortion.py      # 主程序
│   └── matrix_polynomial_math.py  # 数学库
└── tests/
    └── __init__.py
```

## 系统要求

- Python 3.7+
- OpenCV 4.0+
- NumPy 1.19+
- Matplotlib 3.3+

## 使用许可

MIT License

## 作者

[作者信息]

## 更新日志

### v1.0.0 (2024-03-xx)
- 初始版本发布
- 实现基本的畸变校正功能
- 添加可视化对比功能