# BCPS-PF

## Project Overview

This project focuses on a novel particle filtering algorithm: 
the Batch Cyclic Posterior Selection Particle Filter (BCPS-PF). 
We utilize the classic Kitagawa nonlinear system as a validation platform
for a detailed performance evaluation of BCPS-PF. To comprehensively demonstrate
the strengths and characteristics of BCPS-PF, the project also implements and 
compares it against several well-known particle filtering algorithms, including
SIR-PF (Bootstrap PF), EKF-PF (Extended Kalman PF), UPF (Unscented PF), and APF (Auxiliary PF).
Through extensive Monte Carlo simulations, we systematically compare the Root Mean Square Error (RMSE) 
and computational time of these algorithms under varying process noise (Q) and particle counts (N).
The project aims to provide empirical evidence for the effectiveness of the BCPS-PF algorithm and 
explore its potential applications in nonlinear system state estimation.

## File Structure
This project adopts a modular design, ensuring clear and organized code that's easy to understand and extend:
- `main.py`: The main script to run the simulations. It allows you to select different algorithms and configurations.

project_root/
├── src/
│   ├── config.py             # Configuration parameters for simulations and plotting
│   ├── models.py             # Kitagawa model definition and data simulation functions
│   ├── filters.py            # Implementations of various particle filtering algorithms (including BCPS-PF)
│   ├── utils.py              # Helper functions for plotting, result saving, etc.
│   └── __init__.py           # Marks 'src' directory as a Python package
├── main.py                   # Main entry point of the program, coordinating module execution
└── README.md                 # Project documentation file

## Module Descriptions
- `src/config.py`: Centralizes all configurable parameters for simulations (e.g., time steps T, true observation noise R_TRUE, Monte Carlo runs MC_RUNS) and plotting (e.g., line styles, colors). Notably, it includes BCPS-PF specific parameters like BCPS_ALPHA and BCPS_MAX_BATCHES.

- `src/models.py`: Defines the state transition function f_xt and observation function h_zt for the Kitagawa model, along with the simulate_kitagawa function used to generate simulated data (true states and observations).

- `src/filters.py`: This is the core module of the project. It implements not only the comparative algorithms (SIR_PF, EKF_PF, UPF, and APF) but, more importantly, provides a detailed implementation of the BCPS-PF algorithm (BCPS_PF). The shared likelihood function compute_likelihood for all particle filters is also defined here.

- `src/utils.py`: Offers a collection of utility functions primarily used for:

    - Setting Matplotlib plotting styles (setup_plot_style).

    - Generating various result plots, including the BCPS-PF's key metric—the acceptance rate curve (plot_bcps_acceptance_rate), as well as comparative plots for RMSE over time steps (plot_rmse_per_step), RMSE vs. particle count (plot_rmse_vs_particles), and computational time vs. particle count (plot_time_vs_particles).

    - Saving all simulation results (including BCPS-PF performance data) to an Excel file (save_results_to_excel).

- `main.py`: The program's execution entry point. It imports and orchestrates the config, models, filters, and utils modules, executes the Monte Carlo simulations, and calls the relevant functions for saving results and generating plots.

## Installation and Running
1. Clone the repository:
   ```bash
   git clone https://your_repository_url.git
    cd your_project_directory
    ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```
4. Run the main script:
   ```bash
   python main.py
   ```

## Configuration Parameters
All configurable simulation and plotting parameters are centralized within the `SimulationConfig` class
in the `src/config.py` file. You can modify these parameters to adjust the simulation behavior as needed, 
especially the parameters specific to the BCPS-PF algorithm:

```python
# src/config.py
class SimulationConfig:
    # Simulation Parameters
    T = 100               # Number of time steps
    R_TRUE = 1.0          # True observation noise variance
    X0 = 0.1              # Initial true state
    PRIOR_STD = 0.5       # Standard deviation for initial state prior (used to generate initial particles)

    # BCPS-PF Specific Parameters
    BCPS_ALPHA = 0.5      # Alpha parameter for BCPS-PF (controls acceptance probability)
    BCPS_MAX_BATCHES = 50 # Maximum number of batches for BCPS-PF (affects convergence speed)

    MC_RUNS = 200         # Number of Monte Carlo simulation runs
    PARTICLE_COUNTS = [50, 100, 200, 500, 1000] # List of particle counts to test
    Q_VALUES = [0.1, 0.5, 1.0] # List of process noise variance (Q) values to test

    OUTPUT_FOLDER = 'results' # Folder for outputting results

    # Plotting Parameters
    LINE_STYLES = {
        'SIR-PF': '-', 'EKF-PF': '--', 'UPF': ':',
        'BCPS-PF': '-.', # Unique style for BCPS-PF
        'APF': (0, (3, 5, 1, 5)) # dashdotdotted
    }
    COLORS = {
        'SIR-PF': 'blue', 'EKF-PF': 'green', 'UPF': 'red',
        'BCPS-PF': 'purple', # Unique color for BCPS-PF
        'APF': 'orange'
    }
    MARKERS = {
        'SIR-PF': 'o', 'EKF-PF': 's', 'UPF': '^',
        'BCPS-PF': 'D', # Unique marker for BCPS-PF
        'APF': 'X'
    }
    BCPS_PLOT_COLORS = ['blue', 'green', 'red', 'purple', 'orange']
    BCPS_PLOT_MARKERS = ['o', 's', '^', 'D', 'X']
```

## Results

After the program runs, all generated plots and detailed simulation results will be saved in the `results` 
folder at the project's root directory.

- `Plots`: `.png` format performance curves, visually demonstrating BCPS-PF's RMSE, computational time, and its own acceptance rate in comparison to other algorithms under various conditions.

- `Excel File`: `simulation_results.xlsx` contains detailed RMSE and computational time data for each algorithm, as well as the average particle acceptance rate for the BCPS-PF algorithm at each time step.

## Contributing
Contributions to this project are welcome! Feel free to suggest improvements or submit Pull Requests. If you have deeper insights or suggestions regarding the BCPS-PF algorithm, we'd be especially keen to collaborate.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
This project builds upon the foundational work in particle filtering and nonlinear system state estimation. We acknowledge the contributions of the broader research community in this field, which have made this project possible.