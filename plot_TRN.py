import os
# 确保你能正确导入 src/utils.py 中的绘图函数
import sys
from types import SimpleNamespace  # 导入 SimpleNamespace

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import setup_plot_style, plot_xy_rmse_comparison  # 新增的绘图函数

# --- 1. 直接实例化一个配置对象 ---
# 根据你新的算法和需求进行调整
config = SimpleNamespace(
    OUTPUT_FOLDER='results',  # 图片保存路径

    # 假设你的横坐标是时间步，T 是其总长度。如果横坐标是粒子数，那么 PARTICLE_COUNTS 需要匹配
    # 这里为了示例，假设是时间步，你可以根据实际情况调整
    T=100,  # 假设时间步长，用于内部数据处理，如果横坐标是粒子数，这个T就不需要了

    # 算法名称列表，非常重要，必须与Excel中的列名和你的期望完全一致
    ALGORITHM_NAMES=['SIR-PF', 'APF', 'MPF', 'OOSM', 'BCPS-PF'],

    # 绘图的颜色、线型、标记等配置，根据你的5种算法进行调整
    # 确保每个算法都有对应的样式
    LINE_STYLES={
        'SIR-PF': '-',  # 实线
        'APF': '--',  # 虚线
        'MPF': ':',  # 点线
        'OOSM': '-.',  # 点划线
        'BCPS-PF': '-'  # **将'o-'改为'-'，只保留线型**
    },
    COLORS={
        'SIR-PF': 'orangered',
        'APF': 'darkorange',
        'MPF': 'forestgreen',
        'OOSM': 'darkcyan',
        'BCPS-PF': 'royalblue'
    },
    MARKERS={
        'SIR-PF': 'o',  # 圆形标记
        'APF': 's',  # 方形标记
        'MPF': '^',  # 三角形标记
        'OOSM': 'D',  # 菱形标记
        'BCPS-PF': 'X'  # **将'o-'中的'o'移到这里，改为'X'或者你想要的标记**
    }
)

# --- 2. 指定 Excel 文件路径和工作表名称 ---
EXCEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'results', 'simulation_results2.xlsx')  # 假设Excel文件名为 rmse_data.xlsx
EXCEL_SHEET_NAME = 'Avg_RMSE_vs_N'  # 假设数据在 Sheet1

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    print("Starting plotting RMSE data from Excel...")

    # 检查Excel文件是否存在
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"Error: Excel file not found at {EXCEL_FILE_PATH}. Please ensure it exists.")
        sys.exit(1)

    # 确保输出文件夹存在
    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)

    # --- 4. 从 Excel 读取数据并整理 ---
    print(f"Reading data from {EXCEL_FILE_PATH} - Sheet: {EXCEL_SHEET_NAME}...")

    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=EXCEL_SHEET_NAME)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    # 假设第一列是横坐标 (X-axis values)
    x_axis_values = df.iloc[:, 0].values  # 第一列作为横坐标，通常是时间步或粒子数

    # 初始化存储X和Y方向RMSE的字典
    rmse_x_data = {}
    rmse_y_data = {}

    # 读取X方向的RMSE数据 (列索引 1-5 对应 Excel 的 B-F 列)
    # df.iloc[:, 1:6] 对应 Excel 的第 2 到 6 列
    for i, algo_name in enumerate(config.ALGORITHM_NAMES):
        if 1 + i < df.shape[1]:  # 确保列存在
            rmse_x_data[algo_name] = df.iloc[:, 1 + i].values
        else:
            print(f"Warning: Column for X-RMSE of '{algo_name}' not found. Skipping.")

    # 读取Y方向的RMSE数据 (列索引 6-10 对应 Excel 的 G-K 列)
    # df.iloc[:, 6:11] 对应 Excel 的第 7 到 11 列
    for i, algo_name in enumerate(config.ALGORITHM_NAMES):
        if 6 + i < df.shape[1]:  # 确保列存在
            rmse_y_data[algo_name] = df.iloc[:, 6 + i].values
        else:
            print(f"Warning: Column for Y-RMSE of '{algo_name}' not found. Skipping.")

    print("Data loaded and reorganized successfully.")


    # --- 5. 调用新的绘图函数 ---
    setup_plot_style()  # 应用全局绘图样式
    print("Generating RMSE comparison plot...")


    # 调用新的绘图函数
    plot_xy_rmse_comparison(config, x_axis_values, rmse_x_data, rmse_y_data)

    print("All plots generated and saved to 'results' folder.")