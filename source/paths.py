from os.path import join, abspath, dirname

PROJECT_ROOT = join(dirname(dirname(abspath(__file__))))

RESULTS_DIR = join(PROJECT_ROOT, "results")
HMC_POINTS_DIR = join(RESULTS_DIR, "hmc-points")
MH_POINTS_DIR = join(RESULTS_DIR, "mh-points")

PLOTS_DIR = join(RESULTS_DIR, "plots")
HMC_PLOT_PATH = join(PLOTS_DIR, "hmc")
MH_PLOT_PATH = join(PLOTS_DIR, "mh")

HMC_TRACES_DIR = join(PLOTS_DIR, "hmc-traces")
MH_TRACES_DIR = join(PLOTS_DIR, "mh-traces")

HMC_ESS_PATH = join(PLOTS_DIR, "hmc-ess")
MH_ESS_PATH = join(PLOTS_DIR, "mh-ess")

AUSTRIA_TIMELINE_PLOT_PATH = join(PLOTS_DIR, "timeline")

CACHE_DIR = join(PROJECT_ROOT, "cache")
AUSTRIA_DATA_CACHE_PATH = join(CACHE_DIR, "data-austria.csv")

HMC_STATES_DIR = join(CACHE_DIR, "hmc-states")
MH_STATES_DIR = join(CACHE_DIR, "mh-states")

LOGS_DIR = join(PROJECT_ROOT, "logs")
HMC_LOGS_DIR = join(LOGS_DIR, "hmc")
MH_LOGS_DIR = join(LOGS_DIR, "mh")
