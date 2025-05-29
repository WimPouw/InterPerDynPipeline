import pandas as pd
import numpy as np
import glob
import os
import cv2
import math
from bisect import bisect_left
from scipy.signal import savgol_filter

# Path Definitions
INPUT_LAYER1_PATH = '../dataoutput_STEP1_1_rawposedata/'  # Input directory containing tracked keypoint data from STEP1
VIDEO_PATH = "../data_raw/"                          # Raw video files directory
OUTPUT_PATH = '../dataoutput_STEP1_2_timeseries/'     # Output directory for processed timeseries data

# Variable Explanations:
# =====================================================
# Positional Variables:
# - x, y: 2D coordinates in the image frame
# - x_min, x_max, y_min, y_max: Bounding box coordinates for a person
# - centroid_x, centroid_y: Center point of a person's bounding box
# - com_x, com_y: Center of mass for a person (average of upper body keypoint positions)
# - shoulder_midpoint_*: Midpoint between left and right shoulders for each person (p1, p2)
# - wrist_left_*, wrist_right_*: Positions of left and right wrists for each person (p1, p2)
#
# Distance Variables:
# - distance: Euclidean distance between the centroids of both people
# - distance_com: Euclidean distance between centers of mass of both people
# - distance_shoulder_midpoint: Distance between shoulder midpoints of both people
# - *_speed: Euclidean speed of different body parts (calculated from position differences)
# - *_speed_smooth: Smoothed version of speed using Savitzky-Golay filter
#
# Movement Analysis:
# - p1_com_approach_pos, p2_com_approach_pos: Position of each person's COM relative to their
#   initial position, projected onto the connecting line between people. Positive values 
#   indicate movement toward the other person.
#
# Tracking Status:
# - both_tracked: Boolean indicating if both people are tracked in the current frame
# - single_tracked: Boolean indicating if only one person is tracked in the current frame
# =====================================================


def frame_to_time(ts, fps):
    """Convert frame numbers to time in seconds"""
    ts["time"] = [row[1]["frame"] / fps for row in ts.iterrows()]
    return ts


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def assign_left_right(ts):
    """Assign person IDs (0 or 1) based on x-position in the frame"""
    min_x = np.min([val for val in ts['x'] if val != 0])
    max_x = np.max([val for val in ts['x'] if val != 0])
    for index, row in ts.iterrows():
        if row['x'] == 0:  # Skip untracked points
            continue
        if take_closest([min_x, max_x], row['x']) == min_x:
            ts.loc[index, 'person'] = 0  # Left person
        else:
            ts.loc[index, 'person'] = 1  # Right person
    return ts


def process_time_data(ts):
    """Calculate statistics for tracked people using upper body keypoints"""
    ts_upper = ts[ts['keypoint'] < 11]  # Filter to upper body keypoints only
    
    # Calculate stats per person and time
    stats = ts_upper.groupby(['time', 'person']).agg({
        'x': ['min', 'max', 'mean'],
        'y': ['min', 'max', 'mean']
    }).reset_index()
    
    stats.columns = ['time', 'person', 'x_min', 'x_max', 'com_x', 'y_min', 'y_max', 'com_y']
    stats['centroid_x'] = (stats['x_min'] + stats['x_max']) / 2
    stats['centroid_y'] = (stats['y_min'] + stats['y_max']) / 2
    
    return stats


def process_people_data_vectorized(ts, stats):
    """
    Process keypoint data to calculate relative positions and distances.
    Preserves time alignment and handles COM/centroid assignment.
    """
    # Get all unique times from the original data
    all_times = sorted(ts['time'].unique())
    result = pd.DataFrame(index=all_times)
    result.index.name = 'time'
    
    # Process shoulders first (keypoints 5 and 6)
    shoulders = ts[ts['keypoint'].isin([5, 6])].groupby(['time', 'person']).agg({
        'x': 'mean',
        'y': 'mean'
    }).reset_index()
    
    # Process wrists (keypoints 7 and 8)
    wrists = ts[ts['keypoint'].isin([7, 8])].pivot_table(
        index=['time', 'person'],
        columns='keypoint',
        values=['x', 'y']
    ).reset_index()
    wrists.columns = ['time', 'person', 'left_x', 'right_x', 'left_y', 'right_y']
    
    # Get person-specific data
    p0_stats = stats[stats['person'] == 0].set_index('time')
    p1_stats = stats[stats['person'] == 1].set_index('time')
    
    # Process distances where both people exist
    common_times = sorted(set(p0_stats.index) & set(p1_stats.index))
    if common_times:  # Only calculate if we have common times
        result.loc[common_times, 'distance'] = np.sqrt(
            (p0_stats.loc[common_times, 'centroid_x'] - p1_stats.loc[common_times, 'centroid_x'])**2 + 
            (p0_stats.loc[common_times, 'centroid_y'] - p1_stats.loc[common_times, 'centroid_y'])**2
        )
        result.loc[common_times, 'distance_com'] = np.sqrt(
            (p0_stats.loc[common_times, 'com_x'] - p1_stats.loc[common_times, 'com_x'])**2 + 
            (p0_stats.loc[common_times, 'com_y'] - p1_stats.loc[common_times, 'com_y'])**2
        )
    
    # Process shoulders per person
    for person, prefix in [(0, 'p1'), (1, 'p2')]:
        s_person = shoulders[shoulders['person'] == person].set_index('time')
        result[f'shoulder_midpoint_{prefix}_x'] = s_person['x']
        result[f'shoulder_midpoint_{prefix}_y'] = s_person['y']
    
    # Calculate shoulder midpoint distance where possible
    shoulder_cols = ['shoulder_midpoint_p1_x', 'shoulder_midpoint_p2_x',
                    'shoulder_midpoint_p1_y', 'shoulder_midpoint_p2_y']
    shoulder_mask = result[shoulder_cols].notna().all(axis=1)
    if shoulder_mask.any():
        result.loc[shoulder_mask, 'distance_shoulder_midpoint'] = np.sqrt(
            (result.loc[shoulder_mask, 'shoulder_midpoint_p1_x'] - result.loc[shoulder_mask, 'shoulder_midpoint_p2_x'])**2 + 
            (result.loc[shoulder_mask, 'shoulder_midpoint_p1_y'] - result.loc[shoulder_mask, 'shoulder_midpoint_p2_y'])**2
        )
    
    # Process wrists per person and add COM/centroid
    for person, prefix in [(0, 'p1'), (1, 'p2')]:
        w_person = wrists[wrists['person'] == person].set_index('time')
        result[f'wrist_left_{prefix}_x'] = w_person['left_x']
        result[f'wrist_left_{prefix}_y'] = w_person['left_y']
        result[f'wrist_right_{prefix}_x'] = w_person['right_x']
        result[f'wrist_right_{prefix}_y'] = w_person['right_y']
        
        # Use the correct stats object based on person ID
        person_stats = p0_stats if person == 0 else p1_stats
            
        # Add com and centroid with the correct stats object
        result[f'com_{prefix}_x'] = person_stats['com_x']
        result[f'com_{prefix}_y'] = person_stats['com_y']
        result[f'centroid_{prefix}_x'] = person_stats['centroid_x']
        result[f'centroid_{prefix}_y'] = person_stats['centroid_y']

    # Add tracking quality indicators using original data
    person_counts = ts.groupby('time')['person'].nunique()
    result['both_tracked'] = result.index.map(lambda x: person_counts.get(x, 0) == 2)
    result['single_tracked'] = result.index.map(lambda x: person_counts.get(x, 0) == 1)
        
    return result


def calculate_proximity_approach(timeseries_data):
    """
    Calculate each person's position relative to their initial position,
    projecting movement onto the current connecting line between people.
    
    This handles rotation and changing spatial relationships between people.
    Positive values indicate movement toward the other person from initial position.
    
    The reference positions are only established when both people are detected.
    """
    # Create columns for our measurements
    if 'p1_com_approach_pos' not in timeseries_data.columns:
        timeseries_data['p1_com_approach_pos'] = np.nan
    if 'p2_com_approach_pos' not in timeseries_data.columns:
        timeseries_data['p2_com_approach_pos'] = np.nan
    
    # Sort data by time
    sorted_data = timeseries_data.sort_values('time')
    
    # Find first valid frame to establish reference positions
    reference_p1_pos = None
    reference_p2_pos = None
    reference_distance = None
    
    # First pass: find reference frame where both people are detected
    for idx, row in sorted_data.iterrows():
        # Skip rows with NaN values or where both_tracked is False
        if (np.isnan(row['com_p1_x']) or np.isnan(row['com_p1_y']) or 
            np.isnan(row['com_p2_x']) or np.isnan(row['com_p2_y']) or
            ('both_tracked' in row.index and row['both_tracked'] == False)):
            continue
            
        # Get positions for this frame
        p1_pos = np.array([row['com_p1_x'], row['com_p1_y']])
        p2_pos = np.array([row['com_p2_x'], row['com_p2_y']])
        
        # Calculate connecting vector
        connect_vector = p2_pos - p1_pos
        distance = np.linalg.norm(connect_vector)
        
        if distance > 0:
            # We found a valid reference frame
            reference_p1_pos = p1_pos.copy()
            reference_p2_pos = p2_pos.copy()
            reference_distance = distance
            print(f"Reference frame established at time={row['time']}")
            print(f"  Reference p1_pos: {reference_p1_pos}")
            print(f"  Reference p2_pos: {reference_p2_pos}")
            print(f"  Reference distance: {reference_distance}")
            break
    
    if reference_p1_pos is None:
        print("ERROR: Could not establish a reference frame. No valid frames found with both people detected.")
        return timeseries_data
    
    # Second pass: calculate projected positions for all frames
    for idx, row in sorted_data.iterrows():
        # Skip rows with NaN values
        if (np.isnan(row['com_p1_x']) or np.isnan(row['com_p1_y']) or 
            np.isnan(row['com_p2_x']) or np.isnan(row['com_p2_y'])):
            continue
            
        # Get current positions
        p1_pos = np.array([row['com_p1_x'], row['com_p1_y']])
        p2_pos = np.array([row['com_p2_x'], row['com_p2_y']])
        
        # Calculate current connecting vector and direction
        current_connect = p2_pos - p1_pos
        current_distance = np.linalg.norm(current_connect)
        
        # Skip frames where people are at the same position
        if current_distance == 0:
            continue
            
        current_direction = current_connect / current_distance
        
        # Calculate vector from reference position to current position
        p1_vector = p1_pos - reference_p1_pos
        p2_vector = p2_pos - reference_p2_pos
        
        # Project these vectors onto the current connecting line
        # Positive values mean moving toward the other person
        p1_projection = np.dot(p1_vector, current_direction)
        p2_projection = -np.dot(p2_vector, -current_direction)
        
        # Store values
        timeseries_data.loc[idx, 'p1_com_approach_pos'] = p1_projection
        timeseries_data.loc[idx, 'p2_com_approach_pos'] = p2_projection
    
    # Verify results
    filled_p1 = timeseries_data['p1_com_approach_pos'].notna().sum()
    filled_p2 = timeseries_data['p2_com_approach_pos'].notna().sum()
    
    print(f"Frames with filled values - p1: {filled_p1}, p2: {filled_p2}")
    
    return timeseries_data


def main():
    """Main processing function"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Get all CSV files from layer 1 processing
    layer1_files = glob.glob(INPUT_LAYER1_PATH + '*.csv')
    
    # Process each file
    for file_path in layer1_files:
        # Extract video name from file path
        vid_name = os.path.basename(file_path).split('_keypoints_data_layer1.csv')[0]
        print(f"Processing video: {vid_name}")
        
        # Check if already processed
        output_file = f"{OUTPUT_PATH}/{vid_name}_processed_data_layer2.csv"
        if os.path.exists(output_file):
            print("Already processed, skipping...")
            continue
            
        # Determine video format and get FPS
        video_path = None
        for ext in ['.avi', '.mp4', '.mov']:
            if os.path.exists(f"{VIDEO_PATH}/{vid_name}{ext}"):
                video_path = f"{VIDEO_PATH}/{vid_name}{ext}"
                break
                
        if not video_path:
            print(f"Video file not found for {vid_name}")
            continue
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Working on: {vid_name} with fps = {fps}")
        
        # Load and preprocess data
        ts = pd.read_csv(file_path)
        ts = frame_to_time(ts, fps=fps)
        ts = assign_left_right(ts)
        
        # Calculate statistics
        stats = process_time_data(ts)
        
        # Process tracked data
        processed_data = process_people_data_vectorized(ts, stats)
        
        # Create final data frame
        bb_data = pd.merge(
            stats, 
            processed_data.reset_index(), 
            on='time', 
            how='outer'
        )
        
        # Extract time series data (using person 0 as reference)
        timeseries_data = bb_data[bb_data['person'] == 0].copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Remove unnecessary columns
        timeseries_data = timeseries_data.drop(columns='person')
        
        # Calculate desired time step based on FPS
        desired_time_step = 1/fps
        
        # Interpolate missing values
        nan_cols = timeseries_data.columns[timeseries_data.isna().any()].tolist()
        timeseries_data[nan_cols] = timeseries_data[nan_cols].interpolate()
        
        # Smooth distance with Savitzky-Golay filter
        timeseries_data['distance_smooth'] = savgol_filter(timeseries_data['distance'].values, 11, 3)
        
        # Calculate and smooth wrist speeds
        for wrist in ['wrist_left_p1', 'wrist_right_p1', 'wrist_left_p2', 'wrist_right_p2']:
            # Calculate speed as Euclidean distance between consecutive positions
            timeseries_data[f'{wrist}_speed'] = np.sqrt(
                timeseries_data[f'{wrist}_x'].diff()**2 + timeseries_data[f'{wrist}_y'].diff()**2
            )

            # Fill NaN values resulting from diff
            timeseries_data[f'{wrist}_speed'] = timeseries_data[f'{wrist}_speed'].fillna(0)
            
            # Apply Savitzky-Golay filter for smoothing
            timeseries_data[f'{wrist}_speed_smooth'] = savgol_filter(
                timeseries_data[f'{wrist}_speed'].values, 11, 3
            )
        
        # Fill NaN values before calculating proximity approach
        timeseries_data = timeseries_data.fillna(method='ffill')
        
        # Calculate proximity approach
        timeseries_data = calculate_proximity_approach(timeseries_data)
        
        # Fill any remaining NaN values
        timeseries_data = timeseries_data.fillna(method='ffill')
        
        # Smooth proximity approach values
        timeseries_data['p1_com_approach_pos'] = savgol_filter(timeseries_data['p1_com_approach_pos'].values, 11, 3)
        timeseries_data['p2_com_approach_pos'] = savgol_filter(timeseries_data['p2_com_approach_pos'].values, 11, 3)
        
        # Check for any remaining NaN values
        print("Checking for NaN values...")
        nan_counts = timeseries_data.isna().sum()
        if nan_counts.sum() > 0:
            print("Found NaN values after processing:")
            print(nan_counts)
        else:
            print("No NaN values found.")
        
        # Save processed data
        timeseries_data.to_csv(output_file, index=False)
        
        # Check for time continuity
        print("Checking time continuity...")
        time_diffs = np.array([round(val, 2) for val in np.diff(timeseries_data['time'])])
        if not np.all(time_diffs == desired_time_step):
            print(f"Found gaps at times: {np.where(time_diffs != desired_time_step)[0]}")
        else:
            print("No missing time points.")

if __name__ == "__main__":
    main()