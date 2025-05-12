# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.clf() # Clear any existing figures

import pandas as pd
import numpy as np
import os

# Load the data
df = pd.read_csv('./meta/project_point_metadata_ages.csv')

# Convert gender to categorical label
df['gender_label'] = df['gender'].map({0: 'Female', 1: 'Male'})

# Create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios': [1.5, 1]})
fig.suptitle('Mother-Infant Dataset Overview', fontsize=16, fontfamily='serif')

# Define colors for gender
colors = {
    'Female': '#C55A11',   # Orange for females
    'Male': '#4472C4'       # Blue for males
}

# 1. Line plot - Age progression across time points
for idx, row in df.iterrows():
    gender = row['gender_label']
    # Create arrays for days and time points, handling NaN values
    days = []
    timepoints = []

    for i in range(1, 8):   # T1 to T7
        day_col = f'age_T{i}_days'
        if day_col in df.columns and pd.notna(row[day_col]) and row[day_col] > 0:
            days.append(row[day_col])
            timepoints.append(i)

    # Plot the line for this infant
    axs[0, 0].plot(
        timepoints,
        days,
        marker='o',
        color=colors[gender],
        alpha=0.5,
        linewidth=1.5
    )

# Customize the plot
axs[0, 0].set_xlabel('Time Point', fontfamily='serif')
axs[0, 0].set_ylabel('Age (days)', fontfamily='serif')
axs[0, 0].set_title('Infant Age Progression', fontfamily='serif')
axs[0, 0].set_xticks(range(1, 8))
axs[0, 0].set_xticklabels([f'T{i}' for i in range(1, 8)])
axs[0, 0].grid(True, alpha=0.3)

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['Female'], edgecolor='w', label='Female'),
    Patch(facecolor=colors['Male'], edgecolor='w', label='Male')
]
axs[0, 0].legend(handles=legend_elements, loc='upper left')

# 2. Box plot - Age distribution at each time point
time_point_data = []
labels = []

for i in range(1, 8):   # T1 to T7
    col = f'age_T{i}_days'
    if col in df.columns:
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            time_point_data.append(valid_data)
            labels.append(f'T{i}')

# Create violin plots
if time_point_data:
    violin_parts = axs[0, 1].violinplot(
        time_point_data,
        showmeans=True,
        showmedians=True
    )

    # Color the violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor('#7030A0')  # Purple
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

# Set labels
axs[0, 1].set_xticks(range(1, len(labels) + 1))
axs[0, 1].set_xticklabels(labels)
axs[0, 1].set_xlabel('Time Point', fontfamily='serif')
axs[0, 1].set_ylabel('Age (days)', fontfamily='serif')
axs[0, 1].set_title('Age Distribution by Time Point', fontfamily='serif')
axs[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Pie chart - Gender distribution
gender_counts = df['gender_label'].value_counts()
axs[1, 0].pie(
    gender_counts,
    labels=gender_counts.index,
    autopct='%1.1f%%',
    colors=[colors[g] for g in gender_counts.index],
    wedgeprops={'edgecolor': 'w', 'linewidth': 1}
)
axs[1, 0].set_title('Gender Distribution', fontfamily='serif')

# 4. Summary table
# Calculate averages for each time point, handling NaN values
averages_days = []
for i in range(1, 8):
    col = f'age_T{i}_days'
    if col in df.columns:
        avg = df[col].mean()
        if not pd.isna(avg):
            avg_months = avg / 30.44  # Convert to months
            averages_days.append([f"T{i} Average Age", f"{avg:.1f} days ({avg_months:.1f} months)"])

# Count participants at each time point
participants = []
for i in range(1, 8):
    col = f'age_T{i}_days'
    if col in df.columns:
        count = df[col].notna().sum()
        participants.append([f"Participants at T{i}", f"{count}"])

# Overall statistics
summary_data = [
    ["Total Infants", f"{len(df)}"],
    ["Females", f"{(df['gender'] == 0).sum()}"],
    ["Males", f"{(df['gender'] == 1).sum()}"]
]

# Add age ranges if columns exist
if 'age_T1_days' in df.columns:
    summary_data.append(["Age Range T1", f"{df['age_T1_days'].min():.0f}-{df['age_T1_days'].max():.0f} days"])

last_tp = 7
while last_tp > 0:
    col = f'age_T{last_tp}_days'
    if col in df.columns and df[col].notna().sum() > 0:
        summary_data.append([f"Age Range T{last_tp}", f"{df[col].min():.0f}-{df[col].max():.0f} days"])
        break
    last_tp -= 1

# Create the table with the most important summary statistics
axs[1, 1].axis('off')
table = axs[1, 1].table(
    cellText=summary_data + participants,
    loc='center',
    cellLoc='left'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
axs[1, 1].set_title('Dataset Summary', fontfamily='serif')

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure to a file
output_path = "images/mother_infant_analysis.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig) # Explicitly close the figure object
#
#
#
#
#
#
#
#
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
df = pd.read_csv('./meta/project_siblings_metadata_ages_gender.csv')

# Data preprocessing
df['AgeDifference'] = abs(df['P1agedays'] - df['P2agedays'])
df['AverageAge'] = (df['P1agedays'] + df['P2agedays']) / 2
df['GenderCombo'] = df['GenderP1'] + '-' + df['GenderP2']

# Create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios': [1.5, 1]})
fig.suptitle('Sibling Dataset Overview', fontsize=16, fontfamily='serif')

# Define colors for gender combinations
colors = {
    'M-M': '#4472C4',   # Blue
    'M-F': '#7030A0',   # Purple
    'F-M': '#548235',   # Green
    'F-F': '#C55A11'    # Rust/Orange
}

# 1. Scatter plot - P1 age vs P2 age
for gender_combo, group in df.groupby('GenderCombo'):
    axs[0, 0].scatter(
        group['P1agedays'],
        group['P2agedays'],
        color=colors.get(gender_combo, 'gray'),
        alpha=0.7,
        label=gender_combo,
        s=50
    )

# Add reference line
min_age = min(df['P1agedays'].min(), df['P2agedays'].min())
max_age = max(df['P1agedays'].max(), df['P2agedays'].max())
axs[0, 0].plot([min_age, max_age], [min_age, max_age], 'k--', alpha=0.3)
axs[0, 0].set_xlabel('P1 Age (days)', fontfamily='serif')
axs[0, 0].set_ylabel('P2 Age (days)', fontfamily='serif')
axs[0, 0].set_title('Participant Ages (P1 vs P2)', fontfamily='serif')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# 2. Bar chart - Age differences
bar_positions = np.arange(len(df))
bar_width = 0.8
axs[0, 1].bar(
    bar_positions,
    df['AgeDifference'],
    width=bar_width,
    color=[colors.get(combo, 'gray') for combo in df['GenderCombo']]
)
axs[0, 1].set_xticks(bar_positions)
axs[0, 1].set_xticklabels(df['Code'], rotation=90)
axs[0, 1].set_xlabel('Sibling Pair', fontfamily='serif')
axs[0, 1].set_ylabel('Age Difference (days)', fontfamily='serif')
axs[0, 1].set_title('Age Differences by Sibling Pair', fontfamily='serif')
axs[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Pie chart - Gender distribution
gender_counts = df['GenderCombo'].value_counts()
axs[1, 0].pie(
    gender_counts,
    labels=gender_counts.index,
    autopct='%1.1f%%',
    colors=[colors.get(combo, 'gray') for combo in gender_counts.index],
    wedgeprops={'edgecolor': 'w', 'linewidth': 1}
)
axs[1, 0].set_title('Gender Distribution', fontfamily='serif')

# 4. Summary table
summary_data = [
    ["Total Sibling Pairs", f"{len(df)}"],
    ["Average P1 Age", f"{df['P1agedays'].mean()/30:.1f} months"],
    ["Average P2 Age", f"{df['P2agedays'].mean()/30:.1f} months"],
    ["Average Age Difference", f"{df['AgeDifference'].mean()/30:.1f} months"],
    ["Number of Males", f"{(df['GenderP1'] == 'M').sum() + (df['GenderP2'] == 'M').sum()}"],
    ["Number of Females", f"{(df['GenderP1'] == 'F').sum() + (df['GenderP2'] == 'F').sum()}"]
]

# Turn off axis for table
axs[1, 1].axis('off')
table = axs[1, 1].table(
    cellText=[row for row in summary_data],
    colLabels=["Measure", "Value"],
    loc='center',
    cellLoc='left'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
axs[1, 1].set_title('Dataset Summary', fontfamily='serif')

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure to a file
output_path = "images/sibling_analysis.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from ultralytics import YOLO
from pydantic import BaseModel
import cv2
import csv
import numpy as np
import glob
import os
import torch # for gpu support
from itertools import combinations
import sys

torch.cuda.set_device(0)
# Load the model
modelfolder = './model/'
modellocation = glob.glob(modelfolder+"*.pt")[0]
modelfile = os.path.basename(modellocation)
print(f"We are loading in the following YOLO model: {modelfile}")
model = YOLO(modellocation)
# main variables
video_folder = "../data_raw/"
# avi mp4 or other video formats
video_files = glob.glob(video_folder + "*.mp4") + glob.glob(video_folder + "*.avi") + glob.glob(video_folder + "*.mov") + glob.glob(video_folder + "*.mkv")
step1resultfolder = "../datatracked_afterSTEP1/"
print(video_files)

# keypoint names
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

def tensor_to_matrix(results_tensor):
    # this just takes the results output of YOLO and coverts it to a matrix,
    # making it easier to do quick calculations on the coordinates
    results_list = results_tensor.tolist()
    results_matrix = np.matrix(results_list)
    results_matrix[results_matrix==0] = np.nan
    return results_matrix

def check_for_duplication(results):
    # this threshold determines how close two skeletons must be in order to be
    # considered the same person. Arbitrarily chosen for now.
    close_threshold =150
    # missing data tolerance
    miss_tolerance = 0.95 # this means we can miss up to 75% of the keypoints
    drop_indices = []
    if len(results[0].keypoints.xy) > 1:
        conf_scores = []
        # get detection confidence for each skeleton
        for person in tensor_to_matrix(results[0].keypoints.conf):
            conf_scores.append(np.mean(person))
           
        # this list will stores which comparisons need to be made
        combos = list(combinations(range(len(results[0].keypoints.xy)), 2))

        # now loop through these comparisons
        for combo in combos:
            closeness = abs(np.nanmean(tensor_to_matrix(results[0].keypoints.xy[combo[0]]) - 
                        tensor_to_matrix(results[0].keypoints.xy[combo[1]])))
            # if any of them indicate that two skeletons are very close together,
            # we keep the one with higher tracking confidence, and remove the other
            if closeness < close_threshold:
                conf_list = [conf_scores[combo[0]], conf_scores[combo[1]]]
                idx_min = conf_list.index(min(conf_list))
        
                
                drop_indices.append(combo[idx_min])
                
        # additional checks:
        for person in range(len(results[0].keypoints.xy)):
           keypoints_missed =  np.isnan(tensor_to_matrix(results[0].keypoints.xy[person])).sum()/2
           perc_missed = keypoints_missed/len(tensor_to_matrix(results[0].keypoints.xy[person]))
           
           if perc_missed > miss_tolerance:
               drop_indices.append(person)
        
    return list(set(drop_indices))


for video_path in video_files:
    # Video path
    video_path = video_path
    # only if the output is not there yet
    if os.path.exists(step1resultfolder+ os.path.basename(video_path).split('.')[0]+"_annotated_layer1_c150_miss95.mp4"):
        print(f"Output video already exists for {video_path}. Skipping...")
        continue
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
    output_path = step1resultfolder+ vidname+"_annotated_layer1_c150_miss95.mp4"
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
               
        # write empty rows if no person is detected
        if len(results[0].keypoints.xy) == 0:
            csv_writer.writerow([frame_count, None, None, None, None])
        annotated_frame = frame
        
        # only do this if a person is detected
        if len(results[0].keypoints.xy) > 0:
            # Process the results
            drop_indices = []
            
            drop_indices = check_for_duplication(results)
            
            
            for person_idx, person_keypoints in enumerate(results[0].keypoints.xy):
                if person_idx not in drop_indices:
                    colourcode = (0, 255, 0)
                else:
                    colourcode = (255, 0, 0)
                
                for keypoint_idx, keypoint in enumerate(person_keypoints):
                    x, y = keypoint
 
                    # Write to CSV
                    csv_writer.writerow([frame_count, person_idx, keypoint_idx, x.item(), y.item()])
                    
                    # Draw keypoint on the frame
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, colourcode, -1)
                
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
#
#
#
#
#
step1resultfolder = "./datatracked_afterSTEP1/"
results = glob.glob(step1resultfolder + "*.mp4")

# rerender the video to images folder rerendered
# make folder in rerendered
import os
import shutil
import glob

# folder rerendered
rerendered = "./images/rerendered/"
if not os.path.exists(rerendered):
    os.makedirs(rerendered)
# rerender results video using ffmpeg
for video in results:
    # get the name of the video
    video_name = os.path.basename(video)
    # get the name of the video without extension
    video_name_no_ext = os.path.splitext(video_name)[0]
    # create a folder for the video
    video_folder = os.path.join(rerendered, video_name_no_ext)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    # copy the video to the folder
    shutil.copy(video, video_folder)
    # rerender the video using ffmpeg
    os.system(f"ffmpeg -i {video} -vf fps=25 {video_folder}/{video_name_no_ext}.mp4")

#
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import glob as glob

folderstep1output = "./datatracked_afterSTEP1/"

# Load the CSV file
csv_file = glob.glob(folderstep1output + "*.csv")[0]
df = pd.read_csv(csv_file)

# Display the first few rows of the DataFrame
print(df.head())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
