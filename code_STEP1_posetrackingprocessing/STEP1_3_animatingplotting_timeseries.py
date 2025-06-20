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
from moviepy import VideoFileClip

# Path Definitions
INPUT_LAYER1_PATH = '../dataoutput_STEP1_2_timeseries/'  # Input directory containing tracked keypoint data from STEP1
VIDEO_PATH = "../dataoutput_STEP1_1_rawposedata/"                        # Raw video files directory
OUTPUT_PATH = '../dataoutput_STEP1_3_animations/'     # Output directory for processed timeseries data
targetvideo = "point_5_2_kam_5_chair_annotated_layer1_c150_miss95.mp4" # note that sample video must be set to True to process only a sample video
SAMPLE_VIDEO = True # Set to True to process only a sample video, False to process all videos
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# animate only sample video?

if SAMPLE_VIDEO:
    # For sample video, use a specific video file
    allvidsnew =  glob.glob(VIDEO_PATH + "*" + targetvideo)
else:
    allvidsnew = glob.glob(VIDEO_PATH + "*.mp4") + glob.glob(VIDEO_PATH + "*.avi") + glob.glob(VIDEO_PATH + "*.mov") + glob.glob(VIDEO_PATH + "*.mkv")

print(f"Found {len(allvidsnew)} videos to process")

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
    'distance_com': True,
    'shoulder_midpoint_p1_x': False,
    'shoulder_midpoint_p1_y': False,
    'shoulder_midpoint_p2_x': False,
    'shoulder_midpoint_p2_y': False,
    'distance_shoulder_midpoint': False,
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
    'distance_smooth': False,
    'wrist_left_p1_speed': False,
    'wrist_left_p1_speed_smooth': False,
    'wrist_right_p1_speed': False,
    'wrist_right_p1_speed_smooth': False,
    'wrist_left_p2_speed': False,
    'wrist_left_p2_speed_smooth': False,
    'wrist_right_p2_speed': False,
    'wrist_right_p2_speed_smooth': False,
    'p1_com_approach_pos': True,
    'p2_com_approach_pos': True
}

# plotting functions
def plot_timeseries(timeseries_data, current_time=None, figsize=(12, 6)):
    """
    Visualizing of the processed 1_2 motion variable data together with the original video.
    Groups variables by modality with consistent coloring for p1/p2 and left/right.
    
    Parameters:
    -----------
    timeseries_data : pandas.DataFrame
         DataFrame containing all the motion analysis columns
    current_time : float, optional
         current time in seconds to mark with vertical line
    figsize : tuple, optional
         Figure size in inches (width, height) - reduced height for split layout
    """
    # Filter data for only variables marked as True in animate_variables
    active_vars = [var for var, active in animate_variables.items() if active and var in timeseries_data.columns]
    
    if not active_vars:
        print("No variables selected for animation!")
        return None
    
    # Define FIXED color scheme - consistent colors for p1 and p2
    p1_color = '#1f77b4'  # Blue for p1
    p2_color = '#d62728'  # Red for p2
    other_color = '#2ca02c'  # Green for other variables
    
    # Group variables by modality
    distance_vars = [var for var in active_vars if 'distance' in var]
    position_vars = [var for var in active_vars if any(x in var for x in ['_x', '_y', 'approach_pos']) and 'distance' not in var]
    speed_vars = [var for var in active_vars if 'speed' in var]
    
    # Calculate number of subplots needed
    n_plots = sum([len(distance_vars) > 0, len(position_vars) > 0, len(speed_vars) > 0])
    
    if n_plots == 0:
        return None
    
    # Create figure with subplots - use tight layout and high DPI
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False, dpi=100)
    axes = axes.flatten()
    
    # Set consistent style
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    
    plot_idx = 0
    
    # Helper function to get consistent color and style
    def get_style(var_name):
        # Determine person and assign CONSISTENT colors
        if '_p1_' in var_name or 'p1_' in var_name:
            color = p1_color
            person_label = 'P1'
        elif '_p2_' in var_name or 'p2_' in var_name:
            color = p2_color
            person_label = 'P2'
        else:
            color = other_color
            person_label = ''
        
        # Determine side for linestyle - SOLID for left, DASHED for right
        if 'left' in var_name:
            linestyle = '-'  # Solid line for LEFT
            side_label = 'Left'
        elif 'right' in var_name:
            linestyle = '--'  # Dashed line for RIGHT
            side_label = 'Right'
        else:
            linestyle = '-'
            side_label = ''
        
        # Get alpha and linewidth
        if 'smooth' in var_name:
            alpha = 1.0
            linewidth = 3.0  # Thicker for smoothed data
        else:
            alpha = 0.7
            linewidth = 2.0
        
        # Create descriptive label with clear left/right indication
        base_name = var_name.replace('_', ' ').title()
        if person_label and side_label:
            label = f"{person_label} {side_label} {base_name.replace(person_label, '').strip()}"
        elif person_label:
            label = f"{person_label} {base_name.replace(person_label, '').strip()}"
        else:
            label = base_name
        
        return color, linestyle, alpha, linewidth, label
    
    # 1. Distance Plots
    if distance_vars:
        ax = axes[plot_idx]
        for var in distance_vars:
            color, linestyle, alpha, linewidth, label = get_style(var)
            ax.plot(timeseries_data['time'], timeseries_data[var], 
                   label=label, linewidth=linewidth, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_title('Distances Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (pixels)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if 'distance_shoulder_midpoint' in distance_vars or 'distance' in distance_vars:
            ax.invert_yaxis()
        plot_idx += 1
    
    # 2. Position Plots (group p1/p2 and left/right together)
    if position_vars:
        ax = axes[plot_idx]
        for var in position_vars:
            color, linestyle, alpha, linewidth, label = get_style(var)
            ax.plot(timeseries_data['time'], timeseries_data[var], 
                   label=label, linewidth=linewidth, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_title('Positions Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Position', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plot_idx += 1
    
    # 3. Speed Plots (group p1/p2 and left/right together)
    if speed_vars:
        ax = axes[plot_idx]
        for var in speed_vars:
            color, linestyle, alpha, linewidth, label = get_style(var)
            ax.plot(timeseries_data['time'], timeseries_data[var], 
                   label=label, linewidth=linewidth, 
                   color=color, linestyle=linestyle, alpha=alpha)
        
        ax.set_title('Speeds Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (pixels/sec)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plot_idx += 1
    
    # Add vertical line for current time if specified
    if current_time is not None:
        for ax in axes[:plot_idx]:
            ax.axvline(x=current_time, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    # Set common x-label
    axes[plot_idx-1].set_xlabel('Time (seconds)', fontsize=10)
    
    # Improve layout - tighter for split screen
    plt.tight_layout(pad=1.0)
    
    return fig

# Create animation for each video
for vids in allvidsnew:
    vidname = os.path.basename(vids)
    # remove substring "_annotated_layer1"
    lab = "_annotated_layer1"
    vidname = vidname.replace(lab, "")     
    vidname = vidname[:-4]
    # also remove substring "_c150_miss95"
    vidname = vidname.replace("_c150_miss95", "")
    
    print(f"\nProcessing video: {vidname}")
    
    # Check if CSV file exists
    csv_path = os.path.join(INPUT_LAYER1_PATH, f'{vidname}_processed_data_layer2.csv')
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        continue
        
    # Load the CSV file
    timeseries_data = pd.read_csv(csv_path)

    # Output paths
    temp_output = os.path.join(OUTPUT_PATH, f'{vidname}_distance_layer2_temp.mp4')
    final_output = os.path.join(OUTPUT_PATH, f'{vidname}_distance_layer2.mp4')
    
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
    
    # Calculate time step based on FPS
    time_step = 1.0 / fps
    print(f"Time step: {time_step:.4f} seconds per frame")
    
    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    # Create temporary directory for plots
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating animated video...")
        
        # loop over the frames with tqdm progress bar
        current_time = 0.0  # Start at time 0
        for frame_idx in tqdm.tqdm(range(total_frames), desc=f"Processing {vidname}"):
            # read the frame
            success, frame = cap.read()
            if not success:
                break
            
            # plot the timeseries with current time
            fig = plot_timeseries(timeseries_data, current_time)
            if fig is None:
                print("No plot generated, skipping frame")
                out.write(frame)
                current_time += time_step  # Increment by time step
                continue
            
            # save the plot to a temp file with higher DPI
            plot_path = os.path.join(temp_dir, f'plot_{frame_idx:06d}.png')
            fig.savefig(plot_path, dpi=120, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            plt.close(fig)  # Important: close figure to free memory
            
            # Read the plot image
            plot_img = cv2.imread(plot_path)
            if plot_img is None:
                print(f"Error reading plot image: {plot_path}")
                out.write(frame)
                current_time += time_step  # Increment by time step
                continue
            
            # TOP-BOTTOM SPLIT: Video on top, plot on bottom
            # Video on top half
            video_height = height // 2
            frame_resized = cv2.resize(frame, (width, video_height))
            
            # Plot on bottom half
            plot_height = height - video_height
            plot_img_resized = cv2.resize(plot_img, (width, plot_height))
            
            # Create combined frame
            frame_copy = np.zeros((height, width, 3), dtype=np.uint8)
            frame_copy[:video_height, :, :] = frame_resized
            frame_copy[video_height:, :, :] = plot_img_resized
            
            # write the frame to the output video
            out.write(frame_copy)
            current_time += time_step  # Increment by time step (seconds)
        
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