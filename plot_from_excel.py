import os
# 确保你能正确导入 src/utils.py 中的绘图函数
# 假设你的 project_root/ 目录下运行此脚本
# 如果不是，你需要调整 sys.path 或导入方式
import sys

import numpy as np
import pandas as pd

from src.config import SimulationConfig  # 确保正确导入配置类

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import setup_plot_style, plot_rmse_per_step, plot_rmse_vs_particles, \
    plot_time_vs_particles, plot_bcps_acceptance_rate

# --- 1. 直接实例化一个配置对象 ---
# 这个config对象只需要包含绘图函数所需的参数，而不需要模拟相关的参数
config = SimulationConfig()

# --- 2. 指定 Excel 文件路径 ---
EXCEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'results', 'simulation_results.xlsx')

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    print("Starting plotting from Excel data...")

    # 检查Excel文件是否存在
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"Error: Excel file not found at {EXCEL_FILE_PATH}. Please run main.py first to generate results.")
        sys.exit(1)

    # 确保输出文件夹存在
    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)

    # --- 4. 从 Excel 读取数据并重组 ---
    print(f"Reading data from {EXCEL_FILE_PATH}...")
    xls = pd.ExcelFile(EXCEL_FILE_PATH)

    # a. 读取 Avg_RMSE_vs_N
    df_avg_rmse = xls.parse('Avg_RMSE_vs_N', index_col='Number of Particles (N)')
    all_avg_rmse = {}
    for q_value in config.Q_VALUES:
        all_avg_rmse[q_value] = {}
        for algo_name in config.LINE_STYLES.keys():  # 使用任意一个算法列表来迭代名称
            col_name = f'Q={q_value} {algo_name} Avg. RMSE'
            if col_name in df_avg_rmse.columns:
                all_avg_rmse[q_value][algo_name] = df_avg_rmse[col_name].tolist()
            else:
                print(f"Warning: Column '{col_name}' not found in 'Avg_RMSE_vs_N'. Skipping.")

    # b. 读取 Avg_Time_vs_N
    df_avg_time = xls.parse('Avg_Time_vs_N', index_col='Number of Particles (N)')
    all_avg_time = {}
    for q_value in config.Q_VALUES:
        all_avg_time[q_value] = {}
        for algo_name in config.LINE_STYLES.keys():
            col_name = f'Q={q_value} {algo_name} Avg. Time (s)'
            if col_name in df_avg_time.columns:
                all_avg_time[q_value][algo_name] = df_avg_time[col_name].tolist()
            else:
                print(f"Warning: Column '{col_name}' not found in 'Avg_Time_vs_N'. Skipping.")

    # c. 读取 RMSE_Curves_N100
    rmse_curves_N100 = {}
    for q_value in config.Q_VALUES:
        sheet_name = f'RMSE_Curves_N100_Q{str(q_value).replace(".", "_")}'
        if sheet_name in xls.sheet_names:
            df_rmse_curves = xls.parse(sheet_name, index_col='Time Step')
            rmse_curves_N100[q_value] = df_rmse_curves.to_dict('list')
        else:
            print(f"Warning: Sheet '{sheet_name}' not found. Skipping RMSE per step plot for N=100, Q={q_value}.")
            rmse_curves_N100[q_value] = {}

    # d. 读取 BCPS_AcceptRate
    all_avg_accept_rate_bcps = {}
    for q_value in config.Q_VALUES:
        sheet_name = f'BCPS_AcceptRate_Q{str(q_value).replace(".", "_")}'
        if sheet_name in xls.sheet_names:
            df_accept_rate = xls.parse(sheet_name, index_col='Time Step')
            bcps_rate_array = np.zeros((len(config.PARTICLE_COUNTS), config.T))
            for i, N_count in enumerate(config.PARTICLE_COUNTS):
                col_name = f'N={N_count} Avg. Acceptance Rate'
                if col_name in df_accept_rate.columns:
                    bcps_rate_array[i, :] = df_accept_rate[col_name].values
                else:
                    print(
                        f"Warning: Column '{col_name}' not found in '{sheet_name}'. BCPS rate data might be incomplete.")
            all_avg_accept_rate_bcps[q_value] = bcps_rate_array
        else:
            print(f"Warning: Sheet '{sheet_name}' not found. Skipping BCPS acceptance rate plot for Q={q_value}.")
            all_avg_accept_rate_bcps[q_value] = np.zeros((len(config.PARTICLE_COUNTS), config.T))

    print("Data loaded and reorganized successfully.")

    # --- 5. 调用绘图函数 ---
    setup_plot_style()  # 应用全局绘图样式

    print("Generating plots...")
    plot_rmse_vs_particles(config, all_avg_rmse)
    plot_time_vs_particles(config, all_avg_time)
    plot_rmse_per_step(config, rmse_curves_N100)
    plot_bcps_acceptance_rate(config, all_avg_accept_rate_bcps)

    print("All plots generated and saved to 'results' folder.")