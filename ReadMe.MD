# An Open-source Standardized Pipeline for Tracing the Behavioral Dynamics in Social Interactions: Theory-driven Measurements, Analyses, and Archiving (WORK IN PROGRESS)

## Overview
This is a start of the sample code for a Open-source Standardized Pipeline for Tracing the Behavioral Dynamics in Social Interaction. In this repo we have sample data that we can work with to build our pipelines, which will then be rolled out on the full longitudinal dataset.

## Contact
If you want to be added to the github as a collaborator ask wim.pouw@donders.ru.nl with you github username.

## To Do's
- make time series more robust against tracking errors
- add object tracking pipeline, using improved SAM2?
- determine time series features that further need to be extracted
- determine step 3 time series analysis for understanding interpersonal dynamics
- masking and final tracking of the full dataset
- merge (some) steps into single pipeline

## Repository Structure

### Code Directories
```
code_STEP1_posetrackingprocessing/    # Person tracking using YOLO
└── Python code and YOLO models
code_STEP2_trackingtotimeseries/      # Time series processing
└── Python code for generating time series prepared for further analysis and some other features, like generating animations and smoothness calculations
```

### Data Directories
```
data_fulldata_raw/                    # Full dataset storage (empty)
data_fulldata_tracked/                # Full tracking results (filled, but we can leave for now)
data_sampledata_raw/                  # Sample video storage (empty)
data_sampledatatotimeseries_afterSTEP1/  # STEP 1 output: keypoints generated by yolo for top-view
data_sampledatatotimeseries_afterSTEP2/  # STEP 2 output: time series
```

## Time Series Data Documentation

### Temporal Measurements
| Variable | Description |
|----------|-------------|
| `time`   | Time in seconds (based on video frame rate) |
| `frame`  | Frame number |

### Spatial Measurements
#### Distance Metrics Between person 1 and 2
- `distance`: Raw interpersonal distance
- `distance_smooth`: Smoothed distance (One Euro Filter)
- `distance_com`: Center of mass distance
- `distance_shoulder_midpoint`: Distance between shoulder midpoints

#### Position Coordinates
**Shoulder Positions**
- `shoulder_midpoint_p1_x/y`: Person 1 shoulder midpoint
- `shoulder_midpoint_p2_x/y`: Person 2 shoulder midpoint

**Wrist Positions**
- `wrist_left_p1_x/y`: Person 1 left wrist
- `wrist_right_p1_x/y`: Person 1 right wrist
- `wrist_left_p2_x/y`: Person 2 left wrist
- `wrist_right_p2_x/y`: Person 2 right wrist

### Kinematic Measurements
#### Raw Speeds
- `wrist_left_p1_speed`: Person 1 left wrist
- `wrist_right_p1_speed`: Person 1 right wrist
- `wrist_left_p2_speed`: Person 2 left wrist
- `wrist_right_p2_speed`: Person 2 right wrist

#### Smoothed Speeds
- `wrist_left_p1_speed_smooth`: Person 1 left wrist (filtered)
- `wrist_right_p1_speed_smooth`: Person 1 right wrist (filtered)
- `wrist_left_p2_speed_smooth`: Person 2 left wrist (filtered)
- `wrist_right_p2_speed_smooth`: Person 2 right wrist (filtered)

## Processing Pipeline
1. **Video Tracking (STEP 1)**
   - Input: Raw video files
   - Process: YOLO-based person tracking
   - Output: Tracked keypoints data

2. **Time Series Generation (STEP 2)**
   - Input: Tracked keypoints
   - Process: 
     - Distance calculations
     - Speed computations
     - Signal smoothing (One Euro Filter)
   - Output: Processed time series data

3. Time Series analysis (STEP 3) (TODO)
   - Input: Processed time series data

*Note: Due to size and privacy limitations, video files over 100MB are stored somewhere else.*