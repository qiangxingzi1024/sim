# src/config.py

class SimulationConfig:
    """
    Configuration parameters for the Kitagawa model simulation and plotting.
    """

    # --- Simulation Parameters ---
    T = 50               # Number of time steps for the simulation
    R_TRUE = 0.1         # True observation noise variance (R in the Kitagawa model)
    X0 = 5.0              # Initial true state (x_0) for the Kitagawa system
    PRIOR_STD = 2.0      # Standard deviation for generating initial particles (prior distribution)

    # --- BCPS-PF Specific Parameters ---
    BCPS_ALPHA = 0.9      # Alpha parameter for BCPS-PF (controls the acceptance probability/ratio)
    BCPS_MAX_BATCHES = 50 # Maximum number of batches for BCPS-PF (influences convergence speed and computational cost)

    MC_RUNS = 500        # Number of Monte Carlo simulation runs for averaging results
    PARTICLE_COUNTS = [10,30,50,100,300,500] # List of particle counts (N) to test for each algorithm
    Q_VALUES = [1.0,0.1] # List of true process noise variance (Q) values to test

    OUTPUT_FOLDER = 'results' # Folder name to save all simulation outputs (plots, excel files)

    # --- Plotting Parameters ---
    # Line styles for different particle filters in plots
    LINE_STYLES = {
        'SIR-PF': '-',
        'EKF-PF': '-',
        'UPF': '-',
        'BCPS-PF': '-', # Unique style for BCPS-PF
        'APF': '-' # dashdotdotted style for APF
    }
    # Colors for different particle filters in plots
    COLORS = {
        'SIR-PF': 'orangered',
        'EKF-PF': 'forestgreen',
        'UPF': 'darkcyan',
        'BCPS-PF': 'royalblue', # Unique color for BCPS-PF
        'APF': 'darkorange'
    }
    # Markers for different particle filters in plots
    MARKERS = {
        'SIR-PF': 'o',
        'EKF-PF': 's',
        'UPF': '^',
        'BCPS-PF': 'D', # Unique marker for BCPS-PF
        'APF': 'X'
    }
    # Colors specifically for BCPS-PF acceptance rate plots if needed (can be distinct from general colors)
    BCPS_PLOT_COLORS = ['blue', 'green', 'red', 'purple', 'orange']
    # Markers specifically for BCPS-PF acceptance rate plots
    BCPS_PLOT_MARKERS = ['o', 's', '^', 'D', 'X']