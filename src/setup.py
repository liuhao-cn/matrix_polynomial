from setuptools import setup, Extension
import numpy as np
import shutil
import os
import sys

# Determine compiler args based on platform
if sys.platform == 'win32':
    compile_args = ['/openmp']
    link_args = ['/openmp']
else:
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

module = Extension('interpolation_core',
                  sources=['interpolation_core.c'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=compile_args,
                  extra_link_args=link_args)

def copy_extension():
    # 确定扩展名
    ext = '.pyd' if os.name == 'nt' else '.so'
    
    # 查找编译后的文件
    for root, _, files in os.walk('build'):
        for file in files:
            if file.startswith('interpolation_core') and file.endswith(ext):
                src = os.path.join(root, file)
                # 复制到与 matrix_polynomial_math.py 相同的目录
                dst = os.path.join(os.path.dirname(__file__), '..', file)
                shutil.copy2(src, dst)
                print(f"已复制 {src} -> {dst}")
                return
    print("警告：未找到扩展模块文件")

if __name__ == '__main__':
    setup(name='interpolation_core',
          version='1.0',
          ext_modules=[module])
    copy_extension() 