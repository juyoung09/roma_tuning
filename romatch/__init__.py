import os
from .models import roma_outdoor, tiny_roma_v1_outdoor, roma_indoor, roma_outdoor_dinov3

DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1