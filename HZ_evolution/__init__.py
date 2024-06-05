from .hz_utils import *
from .interp_utils import *
from .stat_model import *

import os

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)