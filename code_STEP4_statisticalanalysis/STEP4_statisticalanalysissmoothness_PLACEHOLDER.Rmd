import numpy as np
from scipy.signal import savgol_filter

# Calculate the position of com p1 and com p2, relative to a projected line between two shoulder midpoints
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

# read the metadata TODO
#metadata = pd.read_csv('../data_sampledatatracked_afterSTEP1/metadata.csv')

# loop over the csv layer 1 data
for vids in layeronedat:
    # Load the CSV file
    ts = pd.read_csv(vids)
    #check fps video using opencv2
    vidname = os.path.basename(vids).split('_keypoints_data_layer1.csv')[0]
    print("Processing video: " + vidname)
    # lets check the sampling rate of the original video (its either avi or mp4)
    if os.path.exists(videos + '/' + vidname + '.avi'):
        video_path = videos + '/' + vidname + '.avi'
    elif os.path.exists(videos + '/' + vidname + '.mp4'):
        video_path = videos + '/' + vidname + '.mp4'
    elif os.path.exists(videos + '/' + vidname + '.mov'):
        video_path = videos + '/' + vidname + '.mov'
    else:
        print("Video file not found for " + vidname)
        continue
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    ts = frame_to_time(ts, fps=fps)
    print("working on the following data: " + vidname + " Assuming fps = " + str(fps))
    # check if already processed
    if os.path.exists(outputfol + '/' + vidname + '_processed_data_layer2.csv'):
        print("Already processed, skipping...")
        continue
    # filter the data to only include the two most tracked people
    #ts = identify_main_people(ts)
    ts = assign_left_right(ts)
    # Get valid stats only when both people are tracked
    stats = process_time_data(ts)
        
    # Lets set up a data frame with all the variables
    bb_data = pd.DataFrame(columns=['time', 'person', 'x_min', 'x_max', 'y_min', 'y_max', 'centroid_x', 'centroid_y', 'distance', 'distance_com', 
    'distance_shoulder_midpoint', 'shoulder_midpoint_p1_x', 'shoulder_midpoint_p2_x', 'shoulder_midpoint_p1_y', 'shoulder_midpoint_p2_y',
    'wrist_left_p1_x', 'wrist_left_p1_y', 'wrist_right_p1_x', 'wrist_right_p1_y', 'wrist_left_p2_x', 'wrist_left_p2_y', 'wrist_right_p2_x', 'wrist_right_p2_y', 
    'com_p1_x', 'com_p1_y', 'com_p2_x', 'com_p2_y', 'centroid_p1_x', 'centroid_p1_y', 'centroid_p2_x', 'centroid_p2_y', 'both_tracked', 'single_tracked','p1_com_approach_pos', 'p2_com_approach_pos'])
    
    # Fill time and person data
    for time in ts['time'].unique():
         bb_data.loc[2*time, ('time', 'person')] = [time, 0]
         bb_data.loc[2*time + 1, ('time', 'person')] = [time, 1]

    # VECTORIZED SECTION START
    # Filter upper body points once for all times
    ts_upper = ts[ts['keypoint'] < 11]

    # Process data directly using stats
    processed_data = process_people_data_vectorized(ts, stats)
    
    # Create final bb_data
    bb_data = pd.merge(
         stats, 
         processed_data.reset_index(), 
         on='time', 
         how='outer'
    )

    # Process time series data
    timeseries_data = bb_data[bb_data['person'] == 0] # person is not relevant here anymore so we can drop it
    
    # Interpolate NaN values
    nan_cols = timeseries_data.columns[timeseries_data.isna().any()].tolist()
    timeseries_data[nan_cols] = timeseries_data[nan_cols].interpolate()
        # Add time variables
   # timeseries_data['time'] = timeseries_data['time'] * (1/fps) 
    # Remove unnecessary columns and add variables
    timeseries_data = timeseries_data.drop(columns='person')
    # Now make the time series uniform, as we have some missing person trackings in the data
    # Resample to uniform time steps (e.g., 0.02s)
    desired_time_step = 1/fps
    time_new = np.arange(timeseries_data['time'].min(), timeseries_data['time'].max(), desired_time_step)

    # Interpolate all columns over the new time grid
    #timeseries_data = timeseries_data.set_index('time').reindex(time_new).interpolate().reset_index() # this command makes every value, for each column, repeat after row 5 (ie a flat line for all columns)
    #timeseries_data.rename(columns={'index': 'time'}, inplace=True)
    #timeseries_data = timeseries_data.drop_duplicates(subset='time')  # Remove duplicates
    #timeseries_data = timeseries_data.sort_values('time')  # Ensure sorted order
    # TODO: we also need to save some information about interpolation to check what interpolation were actually doing
    
    # smooth the distance with savitsky golay filter
    timeseries_data['distance_smooth'] = savgol_filter(timeseries_data['distance'].values, 11, 3)
    
    # Apply savgol filter to all wrist speeds
    for wrist in ['wrist_left_p1', 'wrist_right_p1', 'wrist_left_p2', 'wrist_right_p2']:
          # Calculate speed as before
          timeseries_data[f'{wrist}_speed'] = (timeseries_data[f'{wrist}_x'].diff()**2 + 
                                                          timeseries_data[f'{wrist}_y'].diff()**2)**0.5
          
          # apply savgol filter
          timeseries_data[f'{wrist}_speed_smooth'] = savgol_filter(timeseries_data[f'{wrist}_speed'].values, 11, 3)
          
    # now compute the position of com p1 and com p2, relative to a projected line between two shoulder midpoints
    # Fill NaN values before inputting proximity
    timeseries_data = timeseries_data.fillna(method='ffill')
    timeseries_data = calculate_proximity_approach(timeseries_data)
    timeseries_data = timeseries_data.fillna(method='ffill')
    # now smooth the proximity with the same name
    timeseries_data['p1_com_approach_pos'] = savgol_filter(timeseries_data['p1_com_approach_pos'].values, 11, 3)
    timeseries_data['p2_com_approach_pos'] = savgol_filter(timeseries_data['p2_com_approach_pos'].values, 11, 3)
    # Fill NaN values with nearest non-NaN value
    timeseries_data = timeseries_data.fillna(method='ffill')
    # check for NaN values
    print("Checking for NaN values...")
    nan_counts = timeseries_data.isna().sum()
    if nan_counts.sum() > 0:
        print("Found NaN values after processing:")
        print(nan_counts)
    else:
        print("No NaN values found.")
    timeseries_data.to_csv(outputfol + '/' + vidname + '_processed_data_layer2.csv', index=False)
    # Check for missing time points
    print("Checking time continuity...")
    time_diffs = np.array([round(val,2) for val in np.diff(timeseries_data['time'])])
    if not np.all(time_diffs == desired_time_step):
        print(f"Found gaps at times: {np.where(time_diffs != desired_time_step)[0]}")
    else:
        print("No missing time points.")