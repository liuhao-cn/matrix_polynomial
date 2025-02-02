import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build():
    """清理编译产生的临时文件"""
    dirs_to_remove = ['build', 'dist', '*.egg-info']
    files_to_remove = ['*.pyd', '*.so']
    
    for pattern in dirs_to_remove:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f'已删除目录：{path}')
    
    for pattern in files_to_remove:
        for path in Path('.').glob(pattern):
            if path.is_file():
                path.unlink()
                print(f'已删除文件：{path}')

def install_package():
    """安装包"""
    try:
        # 清理旧的构建文件
        clean_build()
        
        # 检查编译器
        if sys.platform == 'win32':
            try:
                subprocess.run(['cl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                print("错误：未找到 MSVC 编译器，请确保已安装 Visual Studio Build Tools")
                return False
        else:
            try:
                subprocess.run(['gcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                print("错误：未找到 GCC 编译器")
                return False
        
        # 安装依赖
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("已安装依赖包")
        
        # 开发模式安装
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
        print("已完成开发模式安装")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"安装过程出错：{e}")
        return False
    except Exception as e:
        print(f"发生未知错误：{e}")
        return False

if __name__ == '__main__':
    if install_package():
        print("\n安装成功！")
        print("可以通过运行以下命令测试安装：")
        print("python src/fix_distortion.py")
    else:
        print("\n安装失败，请查看上述错误信息") 