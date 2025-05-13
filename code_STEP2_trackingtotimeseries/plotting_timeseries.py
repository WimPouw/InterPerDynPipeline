import pandas as pd
import numpy as np
import glob
import os
import cv2
import math
from bisect import bisect_left
from scipy.signal import savgol_filter

# Path Definitions
INPUT_LAYER1_PATH = '../data_tracked_afterSTEP1/'  # Input directory containing tracked keypoint data from STEP1
VIDEO_PATH = "../data_raw/"                        # Raw video files directory
OUTPUT_PATH = '../data_timeseries_afterSTEP2/'     # Output directory for processed timeseries data
