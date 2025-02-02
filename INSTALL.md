# 安装说明

## 方法1：开发模式安装（推荐）

```bash
pip install -e .
```

这种方式适合开发过程中使用，修改代码后不需要重新安装。

## 方法2：正式安装

```bash
pip install .
```

这种方式会将包安装到 Python 的 site-packages 目录。

## 方法3：仅编译 C 扩展

```bash
python setup.py build_ext --inplace
```

这种方式只编译 C 扩展模块，不安装包。

## 系统要求

- Python 3.7+
- C 编译器（Windows 下需要 MSVC，Linux 下需要 GCC）
- OpenMP 支持

## 可能的问题

1. Windows 下编译失败：
   - 确保安装了 Visual Studio Build Tools
   - 确保 Python 和编译器架构匹配（32/64位）

2. Linux 下编译失败：
   - 确保安装了 gcc 和 python-dev
   - 确保安装了 OpenMP 开发包 