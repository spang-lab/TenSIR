from os.path import join, abspath, dirname

PROJECT_ROOT = join(dirname(dirname(abspath(__file__))))

RESULTS_DIR = join(PROJECT_ROOT, "results")
AUSTRIA_MONTHLY_HMC_POINTS_DIR = join(RESULTS_DIR, "austria-hmc-monthly-points")

PLOTS_DIR = join(RESULTS_DIR, "plots")
AUSTRIA_MONTHLY_HMC_PLOT_PATH = join(PLOTS_DIR, "austria-hmc")
AUSTRIA_TIMELINE_PLOT_PATH = join(PLOTS_DIR, "austria-timeline")

CACHE_DIR = join(PROJECT_ROOT, "cache")
AUSTRIA_DATA_CACHE_PATH = join(CACHE_DIR, "data-austria.csv")

HMC_STATES_DIR = join(CACHE_DIR, "hmc-states")

LOGS_DIR = join(PROJECT_ROOT, "logs")
