import pandas as pd
import numpy as np
import glob
import os
import cv2
import math
from bisect import bisect_left
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

# Path Definitions
INPUT_LAYER1_PATH = '../data_tracked_afterSTEP1/'  # Input directory containing tracked keypoint data from STEP1
VIDEO_PATH = "../data_raw/"                        # Raw video files directory
OUTPUT_PATH = '../data_timeseries_afterSTEP2/'     # Output directory for processed timeseries data

# plotting functions
def plot_timeseries(timeseries_data, current_time=None, figsize=(15, 12)):
     """
     Create a comprehensive visualization of motion analysis data.
     
     Parameters:
     -----------
     timeseries_data : pandas.DataFrame
          DataFrame containing all the motion analysis columns
     current_time : int, optional
          time number to mark with vertical line
     figsize : tuple, optional
          Figure size in inches (width, height)
     """
     # Create figure with subplots
     fig = plt.figure(figsize=figsize)
     
     # Define grid layout
     gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.4)
     
     # 1. Distance Plots
     ax1 = fig.add_subplot(gs[0])
     sns.lineplot(data=timeseries_data, x='time', y='distance', label='Raw Distance Centroids', ax=ax1)
     sns.lineplot(data=timeseries_data, x='time', y='distance_com', label='COM Distance', ax=ax1)
     sns.lineplot(data=timeseries_data, x='time', y='distance_shoulder_midpoint', label='Shoulder Midpoint Distance', ax=ax1)
     ax1.set_title('Distances Over Time')
     ax1.set_ylabel('Distance\n(up is closer)')
     ax1.grid(True)
     ax1.invert_yaxis()
     
     # 2. p1_com_approach_pos', 'p2_com_approach_pos
     ax2 = fig.add_subplot(gs[1])
     sns.lineplot(data=timeseries_data, x='time', y='p1_com_approach_pos', label='Proximity Position p1', ax=ax2)
     sns.lineplot(data=timeseries_data, x='time', y='p2_com_approach_pos', label='Proximity Position p2', ax=ax2)
     ax2.set_title('Proximity Positions')
     ax2.set_ylabel('Position')
     ax2.grid(True)
     
     # 3. Wrist Positions
     ax3 = fig.add_subplot(gs[2])
     sns.lineplot(data=timeseries_data, x='time', y='wrist_left_p1_x', label='Left wrist P1 X', ax=ax3)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_right_p1_x', label='Right wrist P1 X', ax=ax3)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_left_p2_x', label='Left wrist P2 X', ax=ax3)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_right_p2_x', label='Right wrist P2 X', ax=ax3)
     ax3.set_title('Wrist X Positions')
     ax3.set_ylabel('Position')
     ax3.grid(True)
     
     # 4. Wrist Speeds
     ax4 = fig.add_subplot(gs[3])
     # Raw speeds
     sns.lineplot(data=timeseries_data, x='time', y='wrist_left_p1_speed', label='Left P1', alpha=0.3, ax=ax4)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_right_p1_speed', label='Right P1', alpha=0.3, ax=ax4)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_left_p2_speed', label='Left P2', alpha=0.3, ax=ax4)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_right_p2_speed', label='Right P2', alpha=0.3, ax=ax4)
     # Smoothed speeds
     sns.lineplot(data=timeseries_data, x='time', y='wrist_left_p1_speed_smooth', label='Left P1 Smooth', ax=ax4)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_right_p1_speed_smooth', label='Right P1 Smooth', ax=ax4)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_left_p2_speed_smooth', label='Left P2 Smooth', ax=ax4)
     sns.lineplot(data=timeseries_data, x='time', y='wrist_right_p2_speed_smooth', label='Right P2 Smooth', ax=ax4)
     ax4.set_title('Wrist Speeds')
     ax4.set_ylabel('Speed')
     ax4.grid(True)
     
     # Add vertical line for current time if specified
     if current_time is not None:
          for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=current_time, color='r', linestyle='--', alpha=0.5)
     
     # Set common x-label
     plt.xlabel('time')
     
     # Adjust layout
     plt.tight_layout()
     
     return fig

import tqdm
import tempfile
import cv2
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# make an annimation
for vids in allvidsnew:
     vidname = os.path.basename(vids)
     # remove substring "_annotated_layer1"
     lab = "_annotated_layer1"
     vidname = vidname.replace(lab, "")     
     vidname = vidname[:-4]
          # Load the CSV file
     timeseries_data = pd.read_csv(inputfol2 + '/' + vidname + '_processed_data_layer2.csv')
     # if already exists, skip
     if os.path.exists(outputfol + '/' + vidname + '_distance_layer2.mp4'):
          print("Already processed, skipping...")
     # load the video file in opencv
     cap = cv2.VideoCapture(vids)
     # Get video properties
     fps = int(cap.get(cv2.CAP_PROP_FPS))
     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     # Define the output video writer
     output_path = inputfol2  + '/' + vidname + '_distance_layer2.mp4'
     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     out = cv2.VideoWriter(outputfol, fourcc, fps, (width, height))
     # loop over the times with tqdm processbar
     time_count = 0
     for _ in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
          # read the time
          success, time = cap.read()
          if not success:
                break
          # plot the distance
          plot = plot_timeseries(timeseries_data, time_count)
          # save the plot to a temp file in the output folder
          with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=tempfol) as f:          
                plot.savefig(f.name)
                slice_start = 2 * (height // 3)
                slice_height = time.shape[0] - slice_start
                plot_img = cv2.imread(f.name)
                # Resize dynamically
                plot_img = cv2.resize(plot_img, (width, slice_height))
                # Assign without shape mismatch
                time[slice_start:slice_start + slice_height, :, :] = plot_img
                # write the time to the output video
                # downsize the video by a half
                out.write(time)
                      
          time_count += 1
     # Release everything
     cap.release()
     out.release()
     print(f"Output video saved as {output_path}")
     # delete all temp files
     for file in glob.glob(outputfol + '/*.png'):
          os.remove(file)