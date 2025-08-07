import os
from pathlib import Path
import sys


PROJECT_PATH = Path(os.path.abspath(__file__)).parent
sys.path.append(str(Path(PROJECT_PATH, 'src')))
DATA_PATH = Path(PROJECT_PATH , 'data')
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)

RESULTS_PATH = Path(PROJECT_PATH, 'results')
if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir(parents=True)

CONFIGS_PATH = Path(PROJECT_PATH, 'configs')
if not CONFIGS_PATH.exists():
    CONFIGS_PATH.mkdir(parents=True)

SOURCE_PATH = Path(PROJECT_PATH, 'src')

VERBOSE = False
VERBOSE_BATCHES=False
DEBUG=False
