import os


def check(dirname):
    """This function creates a directory
    in case it doesn't exist"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


# The project directory
EVAL_DIR = os.path.dirname(os.path.realpath(__file__))

# Folder with datasets
IN_DATASETS_ROOT = check(os.path.join('/Users', 'nejc', 'github', 'msc', 'datasets/'))  #check(os.path.join(EVAL_DIR, 'datasets'))

DATASETS_ROOT = check(os.path.join(EVAL_DIR, 'datasets/'))

# Where the checkpoints are stored
CKPT_ROOT = check(os.path.join(EVAL_DIR, 'archive/'))

# Where the logs are stored
LOGS = check(os.path.join(EVAL_DIR, 'logs/experiments/'))

# Where the algorithms visualizations are dumped
RESULTS_DIR = check(os.path.join(EVAL_DIR, 'results/'))

# Where the imagenet weights are located
INIT_WEIGHTS_DIR = check(os.path.join(EVAL_DIR, 'weights_imagenet/'))

# Where the demo images are located
DEMO_DIR = check(os.path.join(EVAL_DIR, 'demo/'))
