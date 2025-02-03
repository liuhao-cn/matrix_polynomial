# -*- coding: utf-8 -*-
"""
图像畸变校正GUI程序

功能特性：
1. 参数可视化配置
   - 支持6阶多项式系数设置
   - 可配置G矩阵参数
   - 网格参数动态调整

2. 实时仿真功能
   - 畸变/校正效果实时预览
   - 支持参数修改即时生效
   - 噪声水平可调节

3. 配置管理
   - 自动保存/加载配置
   - 支持窗口布局记忆

4. 可视化显示
   - 四窗口对比显示
   - 自适应窗口缩放
   - 结果叠加对比
"""

import ctypes
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import List, Tuple
import matplotlib.font_manager as fm
from fix_distortion import (
    create_grid_image,
    forward_transform,
    backward_transform,
    I_MATRIX,
    G_MATRIX
)
import os

# 设置DPI感知
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ===================== 全局常量 =====================
BOLD_FONT = fm.FontProperties(weight='bold')  # 加粗字体属性
CONFIG_FILE = 'distortion_config.json'        # 配置文件路径
WINDOW_TITLE = "Fix Distortion GUI"           # 窗口标题
DEFAULT_GRID_COLS = 16                        # 默认网格列数
DEFAULT_GRID_ROWS = 12                        # 默认网格行数
DEFAULT_CELL_SIZE = 50                        # 默认单元格大小
DEFAULT_NOISE_LEVEL = 0.1                    # 默认噪声水平
DEFAULT_INTERP_RANGE = 2                     # 默认插值倍数

class FixDistortionGUI:
    """主GUI应用程序类"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        self.root.title(WINDOW_TITLE)
        
        # 设置窗口最大化，并等待完成后再初始化界面
        self.root.state('zoomed')
        self.root.update()  # 确保窗口最大化完成
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置grid权重使得窗口可以跟随缩放
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=3)
        
        # 添加配置文件路径
        self.config_file = CONFIG_FILE
        
        # 创建左侧参数设置区域
        self.params_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="5")
        self.params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # 加载保存的参数
        self.load_config()
        
        # 添加第一次运行标志
        self.first_run = True
        
        # 1. 多项式系数设置（行0-1）
        ttk.Label(self.params_frame, text="多项式系数:").grid(row=0, column=0, sticky=tk.W)
        coef_frame = ttk.Frame(self.params_frame)
        coef_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.poly_coef_I = []  # I系数
        self.poly_coef_G = []  # G系数
        for i in range(6):  # 1到6阶
            ttk.Label(coef_frame, text=f"a{i+1}=").grid(row=i, column=0)
            I_coef = tk.StringVar(value=self.config.get(f'I_coef_{i+1}', "0.0"))
            G_coef = tk.StringVar(value=self.config.get(f'G_coef_{i+1}', "0.0"))
            self.poly_coef_I.append(I_coef)
            self.poly_coef_G.append(G_coef)
            # I项
            entry_I = ttk.Entry(coef_frame, textvariable=I_coef, width=8)
            entry_I.grid(row=i, column=1)
            entry_I.bind('<Return>', lambda e: self.run_simulation())
            label_I = ttk.Label(coef_frame, text="I")
            label_I.grid(row=i, column=2)
            label_I.configure(font=('TkDefaultFont', 10, 'bold'))
            
            # 加号
            ttk.Label(coef_frame, text=" + ").grid(row=i, column=3)
            
            # G项
            entry_G = ttk.Entry(coef_frame, textvariable=G_coef, width=8)
            entry_G.grid(row=i, column=4)
            entry_G.bind('<Return>', lambda e: self.run_simulation())
            label_G = ttk.Label(coef_frame, text="G")
            label_G.grid(row=i, column=5)
            label_G.configure(font=('TkDefaultFont', 10, 'bold'))
        
        # 2. 网格参数设置（行4-5）
        ttk.Label(self.params_frame, text="网格参数:").grid(row=4, column=0, sticky=tk.W)
        self.grid_frame = ttk.Frame(self.params_frame)
        self.grid_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))
        
        # 列数和行数保持原行（row=0）
        ttk.Label(self.grid_frame, text="列数:").grid(row=0, column=0)
        self.grid_cols = tk.StringVar(value=self.config.get('grid_cols', str(DEFAULT_GRID_COLS)))
        grid_cols_entry = ttk.Entry(self.grid_frame, textvariable=self.grid_cols, width=5)
        grid_cols_entry.grid(row=0, column=1)
        grid_cols_entry.bind('<Return>', lambda e: self.run_simulation())
        
        ttk.Label(self.grid_frame, text="行数:").grid(row=0, column=2, padx=(5,0))
        self.grid_rows = tk.StringVar(value=self.config.get('grid_rows', str(DEFAULT_GRID_ROWS)))
        grid_rows_entry = ttk.Entry(self.grid_frame, textvariable=self.grid_rows, width=5)
        grid_rows_entry.grid(row=0, column=3)
        grid_rows_entry.bind('<Return>', lambda e: self.run_simulation())
        
        # 单元格大小移动到新行（row=1）
        ttk.Label(self.grid_frame, text="单元格大小:").grid(row=1, column=0, pady=(5,0))
        self.cell_size = tk.StringVar(value=self.config.get('cell_size', str(DEFAULT_CELL_SIZE)))
        cell_size_entry = ttk.Entry(self.grid_frame, textvariable=self.cell_size, width=5)
        cell_size_entry.grid(row=1, column=1, pady=(5,0))
        cell_size_entry.bind('<Return>', lambda e: self.run_simulation())
        
        # 配置params_frame的列权重
        self.params_frame.grid_columnconfigure(0, weight=1)  # 添加这行使标签列可以占据更多空间
        
        # 3. 噪声水平设置（行7）
        noise_frame = ttk.Frame(self.params_frame)
        noise_frame.grid(row=7, column=0, sticky=(tk.W, tk.E))
        ttk.Label(noise_frame, text="噪声水平:").pack(side=tk.LEFT)
        self.noise_level = tk.StringVar(value=self.config.get('noise_level', str(DEFAULT_NOISE_LEVEL)))
        noise_entry = ttk.Entry(noise_frame, textvariable=self.noise_level, width=8)
        noise_entry.pack(side=tk.LEFT, expand=True)
        noise_entry.bind('<Return>', lambda e: self.run_simulation())
        
        # 4. 插值倍数设置（行8）
        interp_frame = ttk.Frame(self.params_frame)
        interp_frame.grid(row=8, column=0, sticky=(tk.W, tk.E))
        ttk.Label(interp_frame, text="插值倍数:").pack(side=tk.LEFT)
        self.interp_range = tk.StringVar(value=self.config.get('interp_range', str(DEFAULT_INTERP_RANGE)))
        interp_entry = ttk.Entry(interp_frame, textvariable=self.interp_range, width=8)
        interp_entry.pack(side=tk.LEFT, expand=True)
        interp_entry.bind('<Return>', lambda e: self.run_simulation())
        
        # 5. G矩阵设置（行10-11）
        ttk.Label(self.params_frame, text="G矩阵:").grid(row=10, column=0, sticky=tk.W)
        g_matrix_frame = ttk.Frame(self.params_frame)
        g_matrix_frame.grid(row=11, column=0, sticky=(tk.W, tk.E))
        
        self.g_matrix_entries = []
        default_values = self.config.get('G_matrix', [[0, 1], [1, 0]])
        for i in range(2):
            row_entries = []
            for j in range(2):
                var = tk.StringVar(value=str(default_values[i][j]))
                entry = ttk.Entry(g_matrix_frame, textvariable=var, width=8)
                entry.grid(row=i, column=j, padx=2, pady=2)
                entry.bind('<Return>', lambda e: self.run_simulation())
                row_entries.append(var)
            self.g_matrix_entries.append(row_entries)
        
        # 6. 运行按钮（行13）
        run_button_frame = ttk.Frame(self.params_frame)
        run_button_frame.grid(row=13, column=0, sticky=(tk.W, tk.E))
        run_button = ttk.Button(run_button_frame, text="运行", command=self.run_simulation)
        run_button.pack(side=tk.RIGHT, pady=5)
        
        # 创建右侧图形显示区域
        self.fig_frame = ttk.LabelFrame(self.main_frame, text="结果显示", padding="5")
        self.fig_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # 创建画布容器
        self.canvas_frame = ttk.Frame(self.fig_frame)
        self.canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.fig_frame.grid_rowconfigure(0, weight=1)
        self.fig_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 强制更新几何管理器
        self.root.update_idletasks()
        
        # 创建Figure和Canvas，使用实际尺寸
        width = self.canvas_frame.winfo_width()
        height = self.canvas_frame.winfo_height()
        self.fig = Figure(figsize=(width/100, height/100))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建子图
        self.axes = []
        for i in range(2):
            for j in range(2):
                ax = self.fig.add_subplot(2, 2, i*2 + j + 1)
                ax.axis('off')
                self.axes.append(ax)
        
        # 调整布局
        self.fig.tight_layout()
        self.canvas.draw()
        
        # 绑定窗口大小改变事件
        self.canvas_frame.bind('<Configure>', self._on_resize)
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _on_resize(self, event):
        """处理窗口大小改变事件"""
        # 获取当前画布容器的大小
        width = self.canvas_frame.winfo_width()
        height = self.canvas_frame.winfo_height()
        
        # 设置图形大小
        w_inches = width / self.fig.dpi
        h_inches = height / self.fig.dpi
        
        # 更新图形大小并调整布局
        self.fig.set_size_inches(w_inches, h_inches)
        self.fig.tight_layout()
        self.canvas.draw()

    def get_polynomial_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取多项式系数"""
        # 添加常数项1.0作为第一项
        I_coeffs = np.array([float(var.get()) for var in self.poly_coef_I])
        G_coeffs = np.array([float(var.get()) for var in self.poly_coef_G])
        return I_coeffs, G_coeffs

    def get_g_matrix(self) -> np.ndarray:
        """解析G矩阵"""
        try:
            return np.array([[float(self.g_matrix_entries[i][j].get())
                            for j in range(2)] for i in range(2)])
        except:
            messagebox.showerror("错误", "G矩阵格式错误")
            return None

    def parse_range(self, range_str):
        """解析范围字符串为元组"""
        try:
            start, end = map(float, range_str.split(','))
            return start, end
        except:
            messagebox.showerror("错误", "范围格式错误")
            return None

    def run_simulation(self):
        """运行仿真并显示结果"""
        try:
            # 解析参数
            I_coeffs, G_coeffs = self.get_polynomial_coefficients()
            G = self.get_g_matrix()
            if G is None:
                return
                
            # 获取网格参数
            grid_cols = int(self.grid_cols.get())
            grid_rows = int(self.grid_rows.get())
            cell_size = int(self.cell_size.get())
            noise_level = float(self.noise_level.get())
            
            # 获取插值倍数
            interp_range = int(self.interp_range.get())
            
            # 临时修改fix_distortion中的全局参数
            import fix_distortion
            fix_distortion.GRID_COLS = grid_cols
            fix_distortion.GRID_ROWS = grid_rows
            fix_distortion.CELL_SIZE = cell_size
            fix_distortion.NOISE_STD = noise_level
            
            # 修改多项式系数
            fix_distortion.POLY_COEFFS_A = I_coeffs.tolist()  # 设置I系数
            fix_distortion.POLY_COEFFS_B = G_coeffs.tolist()  # 设置G系数
            fix_distortion.G_MATRIX = G  # 设置G矩阵
            
            # 生成网格图像
            original_image = create_grid_image()
            
            # 应用畸变变换
            distorted_image, x_uni, y_uni, u_uni, v_uni = forward_transform(original_image)
            
            # 应用校正变换
            corrected_image = backward_transform(
                distorted_image,
                I_coeffs,
                G_coeffs,
                I=I_MATRIX,
                G=G,
                interp_range=interp_range
            )
            
            # 清除所有子图内容
            for ax in self.axes:
                ax.clear()
                ax.axis('off')
            
            # 显示原始网格
            self.axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
            self.axes[0].set_title('Original Grid', fontsize=9)
            
            # 显示畸变网格
            self.axes[1].imshow(distorted_image, cmap='gray', vmin=0, vmax=255)
            self.axes[1].set_title('Distorted Grid', fontsize=9)
            
            # 显示修正网格
            self.axes[2].imshow(corrected_image, cmap='gray', vmin=0, vmax=255)
            self.axes[2].set_title('Corrected Grid', fontsize=9)
            
            # 显示对比图
            overlay = np.dstack((
                original_image.astype(float)/255,
                corrected_image.astype(float)/255,
                np.zeros_like(original_image)
            ))
            self.axes[3].imshow(overlay)
            self.axes[3].set_title('Comparison, Red=Original, Green=Corrected, Yellow=Identical)', fontsize=9)
            
            # 调整布局并更新画布
            self.fig.tight_layout()
            self.canvas.draw()
            
            # 只在第一次运行时触发窗口刷新
            if self.first_run:
                self.root.after(100, self.refresh_window)
                self.first_run = False
            
        except Exception as e:
            print(f"Error: {str(e)}")
            messagebox.showerror("错误", str(e))

    def _plot_grid(self, ax, X, Y, title='', color='black', alpha=1.0, label=None):
        """绘制网格"""
        # 绘制水平线
        for i in range(X.shape[0]):
            ax.plot(X[i, :], Y[i, :], '-', color=color, alpha=alpha, 
                   label=label if i == 0 else None)
        # 绘制垂直线
        for i in range(X.shape[1]):
            ax.plot(X[:, i], Y[:, i], '-', color=color, alpha=alpha)
        
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')

    def load_config(self):
        """加载配置文件"""
        import json
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {}

    def save_config(self):
        """保存当前配置"""
        import json
        config = {
            'grid_cols': self.grid_cols.get(),
            'grid_rows': self.grid_rows.get(),
            'cell_size': self.cell_size.get(),
            'noise_level': self.noise_level.get(),
            'interp_range': self.interp_range.get(),
            'G_matrix': [[float(self.g_matrix_entries[i][j].get())
                         for j in range(2)] for i in range(2)]
        }
        
        # 保存多项式系数
        for i in range(6):
            config[f'I_coef_{i+1}'] = self.poly_coef_I[i].get()
            config[f'G_coef_{i+1}'] = self.poly_coef_G[i].get()
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {str(e)}")

    def on_closing(self):
        """窗口关闭时的处理"""
        self.save_config()
        self.root.destroy()

    def refresh_window(self):
        """触发窗口缩小和最大化以刷新显示"""
        # 保存当前状态
        current_state = self.root.state()
        
        # 如果当前是最大化状态，先还原再最大化
        if current_state == 'zoomed':
            self.root.state('normal')
            self.root.update()
            self.root.state('zoomed')
        # 如果当前是普通状态，先最大化再还原
        else:
            self.root.state('zoomed')
            self.root.update()
            self.root.state(current_state)

def main():
    root = tk.Tk()
    app = FixDistortionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 