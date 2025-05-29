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
import tqdm
import tempfile
from moviepy.editor import VideoFileClip

# Path Definitions
INPUT_LAYER1_PATH = '../dataoutput_STEP1_2_timeseries/'  # Input directory containing tracked keypoint data from STEP1
VIDEO_PATH = "../data_raw/"                        # Raw video files directory
OUTPUT_PATH = '../dataoutput_STEP1_3_animations/'     # Output directory for processed timeseries data

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# animate only sample video?
SAMPLE_VIDEO = True  # Set to True to process only a sample video, False to process all videos
if SAMPLE_VIDEO:
    # For sample video, use a specific video file
    allvidsnew = [os.path.join(VIDEO_PATH, "sample_annotated_layer1_c150_miss95.mp4")]

# what variables to animate with video
animate_variables = {
    'x_min': False,
    'x_max': False,
    'com_x': False,
    'y_min': False,
    'y_max': False,
    'com_y': False,
    'centroid_x': False,
    'centroid_y': False,
    'distance': False,
    'distance_com': False,
    'shoulder_midpoint_p1_x': False,
    'shoulder_midpoint_p1_y': False,
    'shoulder_midpoint_p2_x': False,
    'shoulder_midpoint_p2_y': False,
    'distance_shoulder_midpoint': True,
    'wrist_left_p1_x': False,
    'wrist_left_p1_y': False,
    'wrist_right_p1_x': False,
    'wrist_right_p1_y': False,
    'com_p1_x': False,
    'com_p1_y': False,
    'centroid_p1_x': False,
    'centroid_p1_y': False,
    'wrist_left_p2_x': False,
    'wrist_left_p2_y': False,
    'wrist_right_p2_x': False,
    'wrist_right_p2_y': False,
    'com_p2_x': False,
    'com_p2_y': False,
    'centroid_p2_x': False,
    'centroid_p2_y': False,
    'distance_smooth': True,
    'wrist_left_p1_speed': False,
    'wrist_left_p1_speed_smooth': True,
    'wrist_right_p1_speed': False,
    'wrist_right_p1_speed_smooth': True,
    'wrist_left_p2_speed': False,
    'wrist_left_p2_speed_smooth': True,
    'wrist_right_p2_speed': False,
    'wrist_right_p2_speed_smooth': True,
    'p1_com_approach_pos': True,
    'p2_com_approach_pos': True
}

# plotting functions
def plot_timeseries(timeseries_data, current_time=None, figsize=(15, 8)):
    """
    Create a comprehensive visualization of motion analysis data.
    Groups variables by modality with consistent coloring for p1/p2 and left/right.
    
    Parameters:
    -----------
    timeseries_data : pandas.DataFrame
         DataFrame containing all the motion analysis columns
    current_time : int, optional
         time number to mark with vertical line
    figsize : tuple, optional
         Figure size in inches (width, height)
    """
    # Filter data for only variables marked as True in animate_variables
    active_vars = [var for var, active in animate_variables.items() if active and var in timeseries_data.columns]
    
    if not active_vars:
        print("No variables selected for animation!")
        return None
    
    # Define consistent color scheme
    # Colors for p1/p2 (blue family for p1, red family for p2)
    p1_colors = {'left': '#1f77b4', 'right': '#aec7e8'}  # Dark blue, light blue
    p2_colors = {'left': '#d62728', 'right': '#ff9896'}  # Dark red, light red
    other_colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Group variables by modality
    distance_vars = [var for var in active_vars if 'distance' in var]
    position_vars = [var for var in active_vars if any(x in var for x in ['_x', '_y', 'approach_pos']) and 'distance' not in var]
    speed_vars = [var for var in active_vars if 'speed' in var]
    
    # Calculate number of subplots needed
    n_plots = sum([len(distance_vars) > 0, len(position_vars) > 0, len(speed_vars) > 0])
    
    if n_plots == 0:
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Helper function to get consistent color and style
    def get_style(var_name):
        # Determine person (p1/p2)
        if '_p1_' in var_name:
            person = 'p1'
        elif '_p2_' in var_name:
            person = 'p2'
        else:
            person = 'other'
        
        # Determine side (left/right)
        if 'left' in var_name:
            side = 'left'
        elif 'right' in var_name:
            side = 'right'
        else:
            side = 'other'
        
        # Get color
        if person == 'p1':
            color = p1_colors.get(side, p1_colors['left'])
        elif person == 'p2':
            color = p2_colors.get(side, p2_colors['left'])
        else:
            color = other_colors[0]  # Use first color for non-person variables
        
        # Get linestyle (solid for left, dashed for right)
        if side == 'right':
            linestyle = '--'
        else:
            linestyle = '-'
        
        # Get alpha (lower for raw data, full for smoothed)
        if 'smooth' in var_name:
            alpha = 1.0
        elif any(x in var_name for x in ['speed', 'velocity']) and 'smooth' not in var_name:
            alpha = 0.3
        else:
            alpha = 1.0
        
        return color, linestyle, alpha
    
    # 1. Distance Plots
    if distance_vars:
        ax = axes[plot_idx]
        color_idx = 0
        for var in distance_vars:
            if any(x in var for x in ['_p1_', '_p2_']):
                color, linestyle, alpha = get_style(var)
            else:
                color = other_colors[color_idx % len(other_colors)]
                linestyle = '-'
                alpha = 1.0
                color_idx += 1
            
            ax.plot(timeseries_data['time'], timeseries_data[var], 
                   label=var.replace('_', ' '), linewidth=2, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_title('Distances Over Time')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if 'distance_shoulder_midpoint' in distance_vars or 'distance' in distance_vars:
            ax.invert_yaxis()
        plot_idx += 1
    
    # 2. Position Plots (group p1/p2 and left/right together)
    if position_vars:
        ax = axes[plot_idx]
        
        for var in position_vars:
            color, linestyle, alpha = get_style(var)
            ax.plot(timeseries_data['time'], timeseries_data[var], 
                   label=var.replace('_', ' '), linewidth=2, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_title('Positions Over Time')
        ax.set_ylabel('Position')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
    # 3. Speed Plots (group p1/p2 and left/right together)
    if speed_vars:
        ax = axes[plot_idx]
        
        for var in speed_vars:
            color, linestyle, alpha = get_style(var)
            ax.plot(timeseries_data['time'], timeseries_data[var], 
                   label=var.replace('_', ' '), linewidth=2, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_title('Speeds Over Time')
        ax.set_ylabel('Speed')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
    # Add vertical line for current time if specified
    if current_time is not None:
        for ax in axes[:plot_idx]:
            ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Set common x-label
    axes[plot_idx-1].set_xlabel('Time (frames)')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Get all video files
allvidsnew = glob.glob(VIDEO_PATH + "/*_annotated_layer1.mp4")
inputfol2 = INPUT_LAYER1_PATH
outputfol = OUTPUT_PATH

print(f"Found {len(allvidsnew)} videos to process")

# Create animation for each video
for vids in allvidsnew:
    vidname = os.path.basename(vids)
    # remove substring "_annotated_layer1"
    lab = "_annotated_layer1"
    vidname = vidname.replace(lab, "")     
    vidname = vidname[:-4]
    
    print(f"\nProcessing video: {vidname}")
    
    # Check if CSV file exists
    csv_path = os.path.join(inputfol2, f'{vidname}_processed_data_layer2.csv')
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        continue
        
    # Load the CSV file
    timeseries_data = pd.read_csv(csv_path)
    
    # Output paths
    temp_output = os.path.join(outputfol, f'{vidname}_distance_layer2_temp.mp4')
    final_output = os.path.join(outputfol, f'{vidname}_distance_layer2.mp4')
    
    # if already exists, skip
    if os.path.exists(final_output):
        print("Already processed, skipping...")
        continue
    
    # load the video file in opencv
    cap = cv2.VideoCapture(vids)
    if not cap.isOpened():
        print(f"Error opening video: {vids}")
        continue
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    # Create temporary directory for plots
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating animated video...")
        
        # loop over the frames with tqdm progress bar
        time_count = 0
        for frame_idx in tqdm.tqdm(range(total_frames), desc=f"Processing {vidname}"):
            # read the frame
            success, frame = cap.read()
            if not success:
                break
            
            # plot the timeseries
            fig = plot_timeseries(timeseries_data, time_count)
            if fig is None:
                print("No plot generated, skipping frame")
                out.write(frame)
                time_count += 1
                continue
            
            # save the plot to a temp file
            plot_path = os.path.join(temp_dir, f'plot_{frame_idx:06d}.png')
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)  # Important: close figure to free memory
            
            # Read the plot image
            plot_img = cv2.imread(plot_path)
            if plot_img is None:
                print(f"Error reading plot image: {plot_path}")
                out.write(frame)
                time_count += 1
                continue
            
            # Calculate overlay area (bottom third of the frame)
            slice_start = 2 * (height // 3)
            slice_height = height - slice_start
            
            # Resize plot to fit overlay area
            plot_img_resized = cv2.resize(plot_img, (width, slice_height))
            
            # Create overlay on the frame
            frame_copy = frame.copy()
            frame_copy[slice_start:slice_start + slice_height, :, :] = plot_img_resized
            
            # write the frame to the output video
            out.write(frame_copy)
            time_count += 1
        
        # Release everything
        cap.release()
        out.release()
        
    print(f"Temporary video saved, now re-encoding with MoviePy...")
    
    # Re-encode with MoviePy for better compatibility
    try:
        clip = VideoFileClip(temp_output)
        clip.write_videofile(final_output, codec='libx264', audio_codec='aac')
        clip.close()
        
        # Remove temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
        print(f"Final output video saved as {final_output}")
        
    except Exception as e:
        print(f"Error during MoviePy re-encoding: {e}")
        print(f"Temporary video available at: {temp_output}")

print("\nAll videos processed!")