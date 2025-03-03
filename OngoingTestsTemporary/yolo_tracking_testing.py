from ultralytics import YOLO
import cv2
import csv
import numpy as np
import glob
import os
import torch
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# GPU setup
torch.cuda.set_device(0)

# Load the model
model = YOLO('yolov8x-pose-p6.pt')

# Main variables
video_folder = "./"
video_files = glob.glob(video_folder + "*.mp4") + glob.glob(video_folder + "*.avi")
output_folder = "./"

# Ensure we have exactly 2 videos
if len(video_files) != 2:
    print(f"Found {len(video_files)} videos. This script requires exactly 2 videos for comparison.")
    print(f"Videos found: {video_files}")
    exit(1)

# Sort videos to ensure consistent ordering
video_files.sort()
video_path1 = video_files[0]
video_path2 = video_files[1]
video_name1 = os.path.basename(video_path1).split('.')[0]
video_name2 = os.path.basename(video_path2).split('.')[0]

print(f"Comparing videos:\n1: {video_path1}\n2: {video_path2}")

class GetKeypoint:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

get_keypoint = GetKeypoint()

# Define skeleton connections
skeleton = [
    (get_keypoint.LEFT_SHOULDER, get_keypoint.RIGHT_SHOULDER),
    (get_keypoint.LEFT_SHOULDER, get_keypoint.LEFT_ELBOW),
    (get_keypoint.RIGHT_SHOULDER, get_keypoint.RIGHT_ELBOW),
    (get_keypoint.LEFT_ELBOW, get_keypoint.LEFT_WRIST),
    (get_keypoint.RIGHT_ELBOW, get_keypoint.RIGHT_WRIST),
    (get_keypoint.LEFT_SHOULDER, get_keypoint.LEFT_HIP),
    (get_keypoint.RIGHT_SHOULDER, get_keypoint.RIGHT_HIP),
    (get_keypoint.LEFT_HIP, get_keypoint.RIGHT_HIP),
    (get_keypoint.LEFT_HIP, get_keypoint.LEFT_KNEE),
    (get_keypoint.RIGHT_HIP, get_keypoint.RIGHT_KNEE),
    (get_keypoint.LEFT_KNEE, get_keypoint.LEFT_ANKLE),
    (get_keypoint.RIGHT_KNEE, get_keypoint.RIGHT_ANKLE),
]

def tensor_to_matrix(results_tensor):
    # Convert YOLO results tensor to matrix for easier calculations
    results_list = results_tensor.tolist()
    results_matrix = np.matrix(results_list)
    results_matrix[results_matrix==0] = np.nan
    return results_matrix

def check_for_duplication(results):
    # Identify duplicate skeletons to be removed
    close_threshold = 350
    miss_tolerance = 0.75
    drop_indices = []
    
    if len(results[0].keypoints.xy) > 1:
        conf_scores = []
        # Get detection confidence for each skeleton
        for person in tensor_to_matrix(results[0].keypoints.conf):
            conf_scores.append(np.mean(person))
                
        # Store which comparisons need to be made
        combos = list(combinations(range(len(results[0].keypoints.xy)), 2))
        
        # Loop through these comparisons
        for combo in combos:
            closeness = abs(np.nanmean(tensor_to_matrix(results[0].keypoints.xy[combo[0]]) - 
                        tensor_to_matrix(results[0].keypoints.xy[combo[1]])))
            # If skeletons are very close, keep the one with higher tracking confidence
            if closeness < close_threshold:
                conf_list = [conf_scores[combo[0]], conf_scores[combo[1]]]
                idx_min = conf_list.index(min(conf_list))
                drop_indices.append(combo[idx_min])
                
        # Additional checks for missing keypoints
        for person in range(len(results[0].keypoints.xy)):
           keypoints_missed = np.isnan(tensor_to_matrix(results[0].keypoints.xy[person])).sum()/2
           perc_missed = keypoints_missed/len(tensor_to_matrix(results[0].keypoints.xy[person]))
           if perc_missed > miss_tolerance:
               drop_indices.append(person)
    
    return list(set(drop_indices))

def process_video(video_path):
    # Process a single video and return keypoints data
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Store keypoints for each frame
    all_keypoints = {}
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
    
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        frame_keypoints = {}
        
        # Process the results if people are detected
        if len(results[0].keypoints.xy) > 0:
            drop_indices = check_for_duplication(results)
            
            for person_idx, person_keypoints in enumerate(results[0].keypoints.xy):
                if person_idx not in drop_indices:
                    person_data = {}
                    for keypoint_idx, keypoint in enumerate(person_keypoints):
                        x, y = keypoint
                        if not (np.isnan(x.item()) or np.isnan(y.item())):
                            person_data[keypoint_idx] = (x.item(), y.item())
                    
                    frame_keypoints[person_idx] = person_data
        
        all_keypoints[frame_count] = frame_keypoints
        frame_count += 1
    
    cap.release()
    return all_keypoints, (width, height, fps, frame_count)

def create_comparison_plot(keypoints1, keypoints2, frame_num, max_frames):
    # Create comparison plots for the current frame
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot for tracking confidence
    tracking_data1 = []
    tracking_data2 = []
    
    # Calculate metrics if data exists
    if frame_num in keypoints1 and keypoints1[frame_num]:
        for person_idx, keypoints in keypoints1[frame_num].items():
            tracking_data1.append(len(keypoints))
    
    if frame_num in keypoints2 and keypoints2[frame_num]:
        for person_idx, keypoints in keypoints2[frame_num].items():
            tracking_data2.append(len(keypoints))
    
    # Keypoints detected comparison
    ax1.bar(['Video 1', 'Video 2'], 
           [sum(tracking_data1)/17 if tracking_data1 else 0, 
            sum(tracking_data2)/17 if tracking_data2 else 0],
           color=['blue', 'orange'])
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Keypoint Detection Completeness')
    ax1.set_ylabel('Ratio of detected keypoints')
    
    # Plot tracking stability over time (last 30 frames)
    history_window = 30
    start_frame = max(0, frame_num - history_window)
    
    detection_history1 = []
    detection_history2 = []
    
    for f in range(start_frame, frame_num + 1):
        # Video 1 detection count
        if f in keypoints1 and keypoints1[f]:
            total_keypoints = sum(len(kps) for kps in keypoints1[f].values())
            max_possible = len(keypoints1[f]) * 17
            detection_history1.append(total_keypoints / max_possible if max_possible > 0 else 0)
        else:
            detection_history1.append(0)
            
        # Video 2 detection count
        if f in keypoints2 and keypoints2[f]:
            total_keypoints = sum(len(kps) for kps in keypoints2[f].values())
            max_possible = len(keypoints2[f]) * 17
            detection_history2.append(total_keypoints / max_possible if max_possible > 0 else 0)
        else:
            detection_history2.append(0)
    
    frames_to_plot = list(range(start_frame, frame_num + 1))
    ax2.plot(frames_to_plot, detection_history1, 'b-', label='Video 1')
    ax2.plot(frames_to_plot, detection_history2, 'orange', label='Video 2')
    ax2.set_xlim(start_frame, frame_num)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Tracking Stability (Past 30 Frames)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Detection Ratio')
    ax2.legend()
    
    # Add frame counter
    fig.suptitle(f'Frame: {frame_num}/{max_frames}', fontsize=14)
    
    # Convert plot to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    plot_image = np.array(canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    # Convert RGBA to RGB
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
    return plot_image

def main():
    # Process both videos
    print(f"Processing video 1: {video_path1}")
    keypoints1, video_info1 = process_video(video_path1)
    width1, height1, fps1, frame_count1 = video_info1
    
    print(f"Processing video 2: {video_path2}")
    keypoints2, video_info2 = process_video(video_path2)
    width2, height2, fps2, frame_count2 = video_info2
    
    # Determine dimensions for the comparison video
    max_frames = min(frame_count1, frame_count2)
    target_height = max(height1, height2)
    combined_width = width1 + width2
    
    # Create output video writer
    output_path = os.path.join(output_folder, f"{video_name1}_vs_{video_name2}_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = min(fps1, fps2)
    
    # Size for the combined frame: videos side by side + space for plot below
    plot_height = 400  # Height reserved for plots
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (combined_width, target_height + plot_height))
    
    # Create CSV for detailed comparison
    csv_path = os.path.join(output_folder, f"{video_name1}_vs_{video_name2}_comparison_data.csv")
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'video1_persons', 'video1_keypoints', 
                            'video2_persons', 'video2_keypoints', 'keypoint_diff'])
    
        # Reopen video captures for rendering
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        
        for frame_idx in range(max_frames):
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not (ret1 and ret2):
                break
                
            # Resize frames to same height if needed
            if height1 != target_height:
                scale = target_height / height1
                frame1 = cv2.resize(frame1, (int(width1 * scale), target_height))
                
            if height2 != target_height:
                scale = target_height / height2
                frame2 = cv2.resize(frame2, (int(width2 * scale), target_height))
            
            # Process keypoints for visualization
            for video_idx, (frame, keypoints) in enumerate([(frame1, keypoints1), (frame2, keypoints2)]):
                if frame_idx in keypoints:
                    for person_idx, person_keypoints in keypoints[frame_idx].items():
                        # Draw keypoints
                        for keypoint_idx, (x, y) in person_keypoints.items():
                            cv2.circle(frame, (int(x), int(y)), 5, 
                                      (0, 255, 0) if video_idx == 0 else (0, 165, 255), -1)
                        
                        # Draw skeleton
                        for connection in skeleton:
                            if connection[0] in person_keypoints and connection[1] in person_keypoints:
                                start_point = tuple(map(int, person_keypoints[connection[0]]))
                                end_point = tuple(map(int, person_keypoints[connection[1]]))
                                cv2.line(frame, start_point, end_point, 
                                        (255, 0, 0) if video_idx == 0 else (255, 0, 165), 2)
            
            # Calculate statistics for CSV
            v1_persons = len(keypoints1.get(frame_idx, {}))
            v1_keypoints = sum(len(kps) for kps in keypoints1.get(frame_idx, {}).values())
            v2_persons = len(keypoints2.get(frame_idx, {}))
            v2_keypoints = sum(len(kps) for kps in keypoints2.get(frame_idx, {}).values())
            keypoint_diff = v1_keypoints - v2_keypoints
            
            # Write to CSV
            csv_writer.writerow([frame_idx, v1_persons, v1_keypoints, 
                                v2_persons, v2_keypoints, keypoint_diff])
            
            # Create comparison plot
            plot_img = create_comparison_plot(keypoints1, keypoints2, frame_idx, max_frames)
            
            # Add video identifiers
            cv2.putText(frame1, f"Video 1: {video_name1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame2, f"Video 2: {video_name2}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Combine frames side by side
            combined_frame = np.zeros((target_height + plot_height, combined_width, 3), dtype=np.uint8)
            combined_frame[:target_height, :width1] = frame1
            combined_frame[:target_height, width1:] = frame2
            
            # Resize plot to fit the width
            plot_img = cv2.resize(plot_img, (combined_width, plot_height))
            combined_frame[target_height:, :] = plot_img
            
            # Write to output video
            out.write(combined_frame)
            
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}/{max_frames}")
        
        # Release resources
        cap1.release()
        cap2.release()
    
    out.release()
    print(f"Comparison video saved as: {output_path}")
    print(f"Comparison data saved as: {csv_path}")

if __name__ == "__main__":
    main()