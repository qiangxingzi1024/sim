# src/utils.py
import os

import matplotlib.pyplot as plt
import pandas as pd


def setup_plot_style():
    """Sets up global Matplotlib plot style for consistent appearance."""
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.style.use('seaborn-v0_8-darkgrid')  # A visually appealing style


def plot_rmse_per_step(config, rmse_curves_N100):
    """
    Plots the RMSE of each algorithm over time steps for N=100 particles.

    Args:
        config (SimulationConfig): Configuration object containing plotting parameters.
        rmse_curves_N100 (dict): Dictionary with RMSE curves for N=100 particles per Q value.
    """
    output_path = os.path.join(config.OUTPUT_FOLDER, 'rmse_per_step_N100.png')

    plt.figure(figsize=(12, 8))

    for q_value in config.Q_VALUES:
        plt.subplot(len(config.Q_VALUES), 1, config.Q_VALUES.index(q_value) + 1)
        for algo_name, rmse_data in rmse_curves_N100[q_value].items():
            plt.plot(rmse_data, label=algo_name,
                     linestyle=config.LINE_STYLES.get(algo_name, '-'),
                     color=config.COLORS.get(algo_name, 'black'),
                     marker=config.MARKERS.get(algo_name, None),
                     markevery=max(1, len(rmse_data) // 10))  # Plot fewer markers for clarity

        plt.title(f'RMSE per Time Step (N=100, Q={q_value})', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.legend(loc='upper right', ncol=2, frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_rmse_vs_particles(config, all_avg_rmse):
    """
    Plots the average RMSE of each algorithm against the number of particles for different Q values.

    Args:
        config (SimulationConfig): Configuration object containing plotting parameters.
        all_avg_rmse (dict): Dictionary with average RMSE data per Q value and particle count.
    """
    output_path = os.path.join(config.OUTPUT_FOLDER, 'avg_rmse_vs_particles.png')

    plt.figure(figsize=(12, 8))

    for q_value in config.Q_VALUES:
        plt.subplot(len(config.Q_VALUES), 1, config.Q_VALUES.index(q_value) + 1)

        for algo_name, rmse_values in all_avg_rmse[q_value].items():
            plt.plot(config.PARTICLE_COUNTS, rmse_values, label=algo_name,
                     linestyle=config.LINE_STYLES.get(algo_name, '-'),
                     color=config.COLORS.get(algo_name, 'black'),
                     marker=config.MARKERS.get(algo_name, 'o'))

        plt.title(f'Avg. RMSE vs. Particle Count (Q={q_value})', fontsize=14)
        plt.xlabel('Number of Particles (N)', fontsize=12)
        plt.ylabel('Avg. RMSE', fontsize=12)
        # plt.xscale('log')  # Use log scale for particle count for better visualization
        plt.xticks(config.PARTICLE_COUNTS, labels=[str(n) for n in config.PARTICLE_COUNTS])
        plt.legend(loc='upper right', ncol=2, frameon=True, shadow=True)
        plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight',facecolor='white')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_time_vs_particles(config, all_avg_time):
    """
    Plots the average computation time of each algorithm against the number of particles.
    Includes an inset plot for specific algorithms across the full N range.

    Args:
        config (SimulationConfig): Configuration object containing plotting parameters.
        all_avg_time (dict): Dictionary with average computation time per Q value and particle count.
    """
    output_path = os.path.join(config.OUTPUT_FOLDER, 'avg_time_vs_particles.png')

    fig, axes = plt.subplots(len(config.Q_VALUES), 1, figsize=(14, 10), dpi=300)
    if len(config.Q_VALUES) == 1:
        axes = [axes]

    plt.rcParams.update({'font.size': 12})  # 基础字体大小

    # 定义内嵌子图需要显示的算法名称
    # 按照你的要求，只显示 SIR, APF, BCPS
    inset_algorithms = ['SIR-PF', 'APF', 'BCPS-PF']

    for q_idx, q_value in enumerate(config.Q_VALUES):
        ax = axes[q_idx]

        for algo_name, time_values in all_avg_time[q_value].items():
            ax.plot(config.PARTICLE_COUNTS, time_values, label=algo_name,
                    linestyle=config.LINE_STYLES.get(algo_name, '-'),
                    color=config.COLORS.get(algo_name, 'black'),
                    marker=config.MARKERS.get(algo_name, 'o'),
                    linewidth=2,
                    markersize=8
                    )

        ax.set_title(f'Avg. Computation Time vs. Particle Count (Q={q_value})', fontsize=16)
        ax.set_xlabel('Number of Particles (N)', fontsize=14)
        ax.set_ylabel('Avg. Time (s)', fontsize=14)
        # ax.set_xscale('log')
        ax.set_xticks(config.PARTICLE_COUNTS)
        ax.set_xticklabels([str(n) for n in config.PARTICLE_COUNTS])
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.legend(loc='upper left', ncol=2, frameon=True, shadow=True, fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.7)

        # Optional: Add an inset plot for selected algorithms across the full N range
        # 这个条件判断不变，仍然只在粒子数量范围广时添加内嵌图
        if len(config.PARTICLE_COUNTS) > 2 and config.PARTICLE_COUNTS[-1] / config.PARTICLE_COUNTS[0] > 10:
            # 调整内嵌图位置和大小
            # 将 x 坐标向左移动，并确保不会太靠近左边缘或图例
            # 尝试放置在主图的左下角或左侧中间，利用空旷区域
            axins = ax.inset_axes([0.15, 0.25, 0.4, 0.45])  # x, y, width, height: 往左下角靠拢，并略微增大
            # x=0.15: 距离左边缘15%
            # y=0.15: 距离下边缘15%
            # width=0.4: 宽度占主图40%
            # height=0.45: 高度占主图45%

            # 设置内嵌图背景色和透明度
            axins.set_facecolor('white')
            axins.patch.set_alpha(0.7)

            # 修改这里：循环指定的算法，并在整个 N 轴上绘制
            for algo_name in inset_algorithms:
                if algo_name in all_avg_time[q_value]:  # 确保算法数据存在
                    time_values = all_avg_time[q_value][algo_name]
                    # 注意：这里不再是 time_values[:inset_points_count]
                    # 而是 time_values 整个列表，表示在整个 N 轴上
                    axins.plot(config.PARTICLE_COUNTS, time_values,
                               linestyle=config.LINE_STYLES.get(algo_name, '-'),
                               color=config.COLORS.get(algo_name, 'black'),
                               marker=config.MARKERS.get(algo_name, 'o'),
                               linewidth=1.5,
                               markersize=6
                               )

            # 内嵌图的 x 轴现在也要是 log 刻度，因为它覆盖了整个 N 范围
            # axins.set_xscale('log')
            axins.set_xticks(config.PARTICLE_COUNTS)  # 刻度与主图保持一致
            axins.set_xticklabels([str(n) for n in config.PARTICLE_COUNTS])  # 刻度标签
            axins.set_title('Zoom In (Selected Algos)', fontsize=10)  # 标题可以更改，更准确
            axins.tick_params(axis='x', labelsize=9)
            axins.tick_params(axis='y', labelsize=9)
            axins.grid(True, which="both", ls=":", alpha=0.6)

            # 为内嵌图添加自己的图例，因为只显示了部分算法，需要明确指出
            # loc='upper left' 或 'upper right' 都可以，根据实际效果选择
            axins.legend(loc='upper left', fontsize=8, frameon=True, shadow=True, ncol=1)

            ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_bcps_acceptance_rate(config, all_avg_accept_rate_bcps):
    """
    Plots the BCPS-PF acceptance rate over time steps for different particle counts and Q values.

    Args:
        config (SimulationConfig): Configuration object containing plotting parameters.
        all_avg_accept_rate_bcps (dict): Dictionary with average acceptance rates for BCPS-PF.
    """
    output_path = os.path.join(config.OUTPUT_FOLDER, 'bcps_acceptance_rate.png')

    plt.figure(figsize=(12, 8))

    for q_idx, q_value in enumerate(config.Q_VALUES):
        plt.subplot(len(config.Q_VALUES), 1, q_idx + 1)

        for N_idx, N_count in enumerate(config.PARTICLE_COUNTS):
            # Ensure the structure matches what's passed from main.py
            # all_avg_accept_rate_bcps[Q_true][bcps_particle_idx, :]
            # So, for a given Q_value, we need to find its corresponding N_idx in config.PARTICLE_COUNTS

            # Use the actual numpy array for a specific Q and N
            accept_rate_data = all_avg_accept_rate_bcps[q_value][N_idx, :]

            plt.plot(accept_rate_data, label=f'N={N_count}',
                     color=config.BCPS_PLOT_COLORS[N_idx % len(config.BCPS_PLOT_COLORS)],
                     marker=config.BCPS_PLOT_MARKERS[N_idx % len(config.BCPS_PLOT_MARKERS)],
                     markevery=max(1, config.T // 10))

        plt.title(f'BCPS-PF Average Acceptance Rate (Q={q_value})', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Acceptance Rate', fontsize=12)
        plt.legend(loc='lower right', ncol=2, frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)  # Acceptance rate is between 0 and 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def save_results_to_excel(config, all_avg_rmse, all_avg_time, rmse_curves_N100, all_avg_accept_rate_bcps):
    """
    Saves all simulation results to an Excel file with multiple sheets.

    Args:
        config (SimulationConfig): Configuration object.
        all_avg_rmse (dict): Average RMSE for different Q and N.
        all_avg_time (dict): Average computation time for different Q and N.
        rmse_curves_N100 (dict): RMSE curves over time for N=100.
        all_avg_accept_rate_bcps (dict): Average BCPS-PF acceptance rates.
    """
    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)

    excel_path = os.path.join(config.OUTPUT_FOLDER, 'simulation_results.xlsx')
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    # Sheet 1: Average RMSE vs. N
    df_avg_rmse = pd.DataFrame(index=config.PARTICLE_COUNTS)
    for q_value in config.Q_VALUES:
        for algo_name, rmse_values in all_avg_rmse[q_value].items():
            col_name = f'Q={q_value} {algo_name} Avg. RMSE'
            df_avg_rmse[col_name] = rmse_values
    df_avg_rmse.index.name = 'Number of Particles (N)'
    df_avg_rmse.to_excel(writer, sheet_name='Avg_RMSE_vs_N')

    # Sheet 2: Average Time vs. N
    df_avg_time = pd.DataFrame(index=config.PARTICLE_COUNTS)
    for q_value in config.Q_VALUES:
        for algo_name, time_values in all_avg_time[q_value].items():
            col_name = f'Q={q_value} {algo_name} Avg. Time (s)'
            df_avg_time[col_name] = time_values
    df_avg_time.index.name = 'Number of Particles (N)'
    df_avg_time.to_excel(writer, sheet_name='Avg_Time_vs_N')

    # Sheet 3: RMSE Curves for N=100
    for q_value in config.Q_VALUES:
        df_rmse_curves = pd.DataFrame(rmse_curves_N100[q_value])
        df_rmse_curves.index.name = 'Time Step'
        df_rmse_curves.to_excel(writer, sheet_name=f'RMSE_Curves_N100_Q{str(q_value).replace(".", "_")}')

    # Sheet 4: BCPS-PF Acceptance Rates
    for q_value in config.Q_VALUES:
        df_accept_rate = pd.DataFrame(index=range(config.T))
        for N_idx, N_count in enumerate(config.PARTICLE_COUNTS):
            col_name = f'N={N_count} Avg. Acceptance Rate'
            df_accept_rate[col_name] = all_avg_accept_rate_bcps[q_value][N_idx, :]
        df_accept_rate.index.name = 'Time Step'
        df_accept_rate.to_excel(writer, sheet_name=f'BCPS_AcceptRate_Q{str(q_value).replace(".", "_")}')

    writer._save()  # Use _save() for newer pandas versions
    print(f"All simulation results saved to {excel_path}")