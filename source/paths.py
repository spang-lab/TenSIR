from os.path import join, abspath, dirname

PROJECT_ROOT = join(dirname(dirname(abspath(__file__))))

RESULTS_DIR = join(PROJECT_ROOT, "results")

PLOTS_DIR = join(RESULTS_DIR, "plots")

CACHE_DIR = join(PROJECT_ROOT, "cache")
AUSTRIA_DATA_CACHE_PATH = join(CACHE_DIR, "data-austria.csv")
