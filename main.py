# main.py
import os

import numpy as np

# Import modules from src package
from src.config import SimulationConfig
from src.filters import SIR_PF, EKF_PF, UPF, BCPS_PF, APF  # Import all filter functions
from src.models import simulate_kitagawa  # Only need the simulation function here
from src.utils import (
    setup_plot_style,
    plot_rmse_per_step,
    plot_rmse_vs_particles,
    plot_time_vs_particles,
    plot_bcps_acceptance_rate,
    save_results_to_excel
)


def run_monte_carlo_simulations(config: SimulationConfig) -> tuple[dict, dict, dict, dict]:
    """
    Runs Monte Carlo simulations to compare different particle filtering algorithms
    under various noise parameters and particle counts.

    Args:
        config (SimulationConfig): An instance of the SimulationConfig class
                                   containing all simulation parameters.

    Returns:
        tuple:
            - all_avg_rmse (dict): Average RMSE for each algorithm across different Q and N.
            - all_avg_time (dict): Average computation time for each algorithm across different Q and N.
            - all_rmse_curves_N100 (dict): RMSE curves over time for N=100 particles.
            - all_avg_accept_rate_bcps (dict): Average BCPS-PF acceptance rates over time.
    """
    # Initialize dictionaries to store results
    all_avg_rmse = {q: {} for q in config.Q_VALUES}
    all_avg_time = {q: {} for q in config.Q_VALUES}
    all_rmse_curves_N100 = {q: {} for q in config.Q_VALUES}

    # Store BCPS-PF acceptance rates. Initialize as a NumPy array for easier averaging.
    # Shape: (num_Q_values, num_N_values, T) - this will be handled within the loop per Q
    all_avg_accept_rate_bcps = {q: np.zeros((len(config.PARTICLE_COUNTS), config.T)) for q in config.Q_VALUES}

    # Ensure output directory exists
    if not os.path.exists(config.OUTPUT_FOLDER):
        os.makedirs(config.OUTPUT_FOLDER)

    # Loop through different process noise (Q) values
    for q_idx, Q_true in enumerate(config.Q_VALUES):
        print(f"\n===== Starting simulations for Process Noise Variance Q = {Q_true} =====")

        # Lists to store average RMSE and time for current Q, across different N
        avg_rmse_sir_N, avg_rmse_epf_N, avg_rmse_upf_N, avg_rmse_bcps_N, avg_rmse_apf_N = [], [], [], [], []
        avg_time_sir_N, avg_time_epf_N, avg_time_upf_N, avg_time_bcps_N, avg_time_apf_N = [], [], [], [], []

        # This index helps map PARTICLE_COUNTS to the correct row in all_avg_accept_rate_bcps array
        bcps_particle_row_idx = 0

        # Loop through different particle counts (N)
        for current_N in config.PARTICLE_COUNTS:
            print(f"--- Particle Count N = {current_N}, Running {config.MC_RUNS} Monte Carlo runs ---")

            # Lists to store RMSE and time for each MC run
            rmse_sir_runs, rmse_epf_runs, rmse_upf_runs, rmse_bcps_runs, rmse_apf_runs = [], [], [], [], []
            time_sir_runs, time_epf_runs, time_upf_runs, time_bcps_runs, time_apf_runs = [], [], [], [], []

            # Store squared errors over time for RMSE curve calculation
            errors_sir = np.zeros((config.MC_RUNS, config.T))
            errors_epf = np.zeros((config.MC_RUNS, config.T))
            errors_upf = np.zeros((config.MC_RUNS, config.T))
            errors_bcps = np.zeros((config.MC_RUNS, config.T))
            errors_apf = np.zeros((config.MC_RUNS, config.T))

            # Store acceptance rates for BCPS-PF
            accept_rate_list_bcps_runs = np.zeros((config.MC_RUNS, config.T))

            # Run Monte Carlo simulations
            for run_num in range(config.MC_RUNS):
                if (run_num + 1) % (max(1, config.MC_RUNS // 10)) == 0:  # Print progress every 10%
                    print(f'  Q={Q_true}, N={current_N}: Completed {run_num + 1}/{config.MC_RUNS} runs')

                seed = run_num + 1  # Use a distinct seed for each run for reproducibility
                x_true, y_obs = simulate_kitagawa(config.T, Q_true, config.R_TRUE, config.X0, seed=seed)

                # Run SIR-PF
                x_sir_est, _, t_sir = SIR_PF(y_obs, current_N, Q_true, config.R_TRUE, config.X0, config.PRIOR_STD)
                errors_sir[run_num, :] = (x_sir_est - x_true) ** 2
                rmse_sir_runs.append(np.sqrt(np.mean((x_true - x_sir_est) ** 2)))
                time_sir_runs.append(t_sir)

                # Run EPF
                x_epf_est, _, t_epf = EKF_PF(y_obs, current_N, Q_true, config.R_TRUE, config.X0, P0=1.0)
                errors_epf[run_num, :] = (x_epf_est - x_true) ** 2
                rmse_epf_runs.append(np.sqrt(np.mean((x_true - x_epf_est) ** 2)))
                time_epf_runs.append(t_epf)

                # Run UPF
                x_upf_est, _, t_upf = UPF(y_obs, current_N, Q_true, config.R_TRUE, config.X0, P0=1.0)
                errors_upf[run_num, :] = (x_upf_est - x_true) ** 2
                rmse_upf_runs.append(np.sqrt(np.mean((x_true - x_upf_est) ** 2)))
                time_upf_runs.append(t_upf)

                # Run BCPS-PF
                x_bcps_est, _, t_bcps, accept_rate_bcps = BCPS_PF(
                    y_obs, current_N, Q_true, config.R_TRUE, config.X0,
                    alpha=config.BCPS_ALPHA, prior_std=config.PRIOR_STD,
                    max_batches=config.BCPS_MAX_BATCHES
                )
                errors_bcps[run_num, :] = (x_bcps_est - x_true) ** 2
                accept_rate_list_bcps_runs[run_num, :] = accept_rate_bcps
                rmse_bcps_runs.append(np.sqrt(np.mean((x_true - x_bcps_est) ** 2)))
                time_bcps_runs.append(t_bcps)

                # Run APF
                x_apf_est, _, t_apf = APF(y_obs, current_N, Q_true, config.R_TRUE, config.X0, config.PRIOR_STD)
                errors_apf[run_num, :] = (x_apf_est - x_true) ** 2
                rmse_apf_runs.append(np.sqrt(np.mean((x_true - x_apf_est) ** 2)))
                time_apf_runs.append(t_apf)

            # Store average RMSE and time for current N
            avg_rmse_sir_N.append(np.mean(rmse_sir_runs))
            avg_rmse_epf_N.append(np.mean(rmse_epf_runs))
            avg_rmse_upf_N.append(np.mean(rmse_upf_runs))
            avg_rmse_bcps_N.append(np.mean(rmse_bcps_runs))
            avg_rmse_apf_N.append(np.mean(rmse_apf_runs))

            avg_time_sir_N.append(np.mean(time_sir_runs))
            avg_time_epf_N.append(np.mean(time_epf_runs))
            avg_time_upf_N.append(np.mean(time_upf_runs))
            avg_time_bcps_N.append(np.mean(time_bcps_runs))
            avg_time_apf_N.append(np.mean(time_apf_runs))

            # Store average acceptance rate for BCPS-PF for the current Q and N
            all_avg_accept_rate_bcps[Q_true][bcps_particle_row_idx, :] = np.mean(accept_rate_list_bcps_runs, axis=0)
            bcps_particle_row_idx += 1  # Move to the next row for the next N

            # If current_N is 100, store the RMSE curves over time
            if current_N == 100:
                all_rmse_curves_N100[Q_true]['SIR-PF'] = np.sqrt(np.mean(errors_sir, axis=0))
                all_rmse_curves_N100[Q_true]['EPF'] = np.sqrt(np.mean(errors_epf, axis=0))
                all_rmse_curves_N100[Q_true]['UPF'] = np.sqrt(np.mean(errors_upf, axis=0))
                all_rmse_curves_N100[Q_true]['APF'] = np.sqrt(np.mean(errors_apf, axis=0))
                all_rmse_curves_N100[Q_true]['BCPS-PF'] = np.sqrt(np.mean(errors_bcps, axis=0))


        # Store all average RMSE and time for the current Q value
        all_avg_rmse[Q_true]['SIR-PF'] = avg_rmse_sir_N
        all_avg_rmse[Q_true]['EPF'] = avg_rmse_epf_N
        all_avg_rmse[Q_true]['UPF'] = avg_rmse_upf_N
        all_avg_rmse[Q_true]['APF'] = avg_rmse_apf_N
        all_avg_rmse[Q_true]['BCPS-PF'] = avg_rmse_bcps_N


        all_avg_time[Q_true]['SIR-PF'] = avg_time_sir_N
        all_avg_time[Q_true]['EPF'] = avg_time_epf_N
        all_avg_time[Q_true]['UPF'] = avg_time_upf_N
        all_avg_time[Q_true]['APF'] = avg_time_apf_N
        all_avg_time[Q_true]['BCPS-PF'] = avg_time_bcps_N


    return all_avg_rmse, all_avg_time, all_rmse_curves_N100, all_avg_accept_rate_bcps


if __name__ == "__main__":
    # 1. Create simulation configuration instance
    sim_config = SimulationConfig()

    print("--- Particle Filtering Algorithm Performance Comparison Simulation Started ---")

    # 2. Run Monte Carlo simulations
    # This function extracts the core simulation logic.
    avg_rmse, avg_time, rmse_curves_N100, avg_accept_rate_bcps = run_monte_carlo_simulations(sim_config)

    print("\n--- Simulation complete, starting to save results and generate plots ---")

    # 3. Set up plot style
    setup_plot_style()

    # 4. Save results to Excel file
    save_results_to_excel(sim_config, avg_rmse, avg_time, rmse_curves_N100, avg_accept_rate_bcps)

    # 5. Generate and save plots
    plot_rmse_per_step(sim_config, rmse_curves_N100)
    plot_rmse_vs_particles(sim_config, avg_rmse)
    plot_time_vs_particles(sim_config, avg_time)
    plot_bcps_acceptance_rate(sim_config, avg_accept_rate_bcps)

    print("\n--- All results saved and plots generated ---")