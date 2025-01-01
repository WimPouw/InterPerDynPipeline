from ultralytics import YOLO
from pydantic import BaseModel
import cv2
import csv
import numpy as np
import glob
import os
import torch # for gpu support

torch.cuda.set_device(0)
# Load the model
model = YOLO('yolov8x-pose-p6.pt') # heaviest model
# main variables
video_folder = "../data_sampledata_raw/"
# avi mp4 or other video formats
video_files = glob.glob(video_folder + "*.mp4") + glob.glob(video_folder + "*.avi")
step1resultfolder = "../step1result/" # we can replace with full data
print(video_files)

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

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

for video_path in video_files:
    # Video path
    video_path = video_path
    # vidname without extension
    vidname = os.path.basename(video_path)
    vidname = vidname.split('.')[0]

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    corename = os.path.basename(video_path).split('.')[0]
    output_path = step1resultfolder+ vidname+"_annotated_layer1.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Prepare CSV file
    csv_path = step1resultfolder+ vidname+'_keypoints_data_layer1.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write header
    header = ['frame', 'person', 'keypoint', 'x', 'y']
    csv_writer.writerow(header)

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # write empty rows if no person is detected
        if len(results[0].keypoints.xy) == 0:
            csv_writer.writerow([frame_count, None, None, None, None])
            annotated_frame = frame
        
        # only do this if a person is detected
        if len(results[0].keypoints.xy) > 0:
            # Process the results
            for person_idx, person_keypoints in enumerate(results[0].keypoints.xy):
                for keypoint_idx, keypoint in enumerate(person_keypoints):
                    x, y = keypoint
                    # Write to CSV
                    csv_writer.writerow([frame_count, person_idx, keypoint_idx, x.item(), y.item()])
                    
                    # Draw keypoint on the frame
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
                # Draw skeleton
                for connection in skeleton:
                    if connection[0] < len(person_keypoints) and connection[1] < len(person_keypoints):
                        start_point = tuple(map(int, person_keypoints[connection[0]]))
                        end_point = tuple(map(int, person_keypoints[connection[1]]))
                        if all(start_point) and all(end_point):  # Check if both points are valid
                            cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)
        
        # Write the frame to the output video
        out.write(annotated_frame)
        
        frame_count += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Output video saved as {output_path}")
    print(f"Keypoints data saved as {csv_path}")