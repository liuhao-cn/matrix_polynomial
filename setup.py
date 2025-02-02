from setuptools import setup, Extension, find_packages
import numpy as np
import sys
import os

# Determine compiler args based on platform
if sys.platform == 'win32':
    compile_args = ['/O2']  # MSVC优化选项
    link_args = []
else:
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

# Define the extension module
interpolation_core = Extension(
    'interpolation_core',
    sources=['src/interpolation_core.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args
)

setup(
    name='distortion_correction',
    version='1.0.0',
    description='A polynomial-based image distortion correction system',
    author='[Your Name]',
    author_email='[Your Email]',
    packages=find_packages(),
    ext_modules=[interpolation_core],
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.0.0',
        'matplotlib>=3.3.0',
    ],
    python_requires='>=3.7',
) 