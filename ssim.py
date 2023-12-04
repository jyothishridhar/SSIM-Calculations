import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim

# Rest of the code is the same

def calculate_ssim_for_each_frame(distorted_video_path, good_video_path):
    # Open the videos
    distorted_video = cv2.VideoCapture(distorted_video_path)
    good_video = cv2.VideoCapture(good_video_path)

    ssim_values = []
    distorted_frame_numbers = []
    frame_timestamps = []  # To store the timestamp of each frame

    while True:
        # Read frames from both videos
        ret1, distorted_frame = distorted_video.read()
        ret2, good_frame = good_video.read()

        # If frames are not retrieved, the videos have ended
        if not ret1 or not ret2:
            break

        # Convert frames to grayscale
        distorted_frame_gray = cv2.cvtColor(distorted_frame, cv2.COLOR_BGR2GRAY)
        good_frame_gray = cv2.cvtColor(good_frame, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM for the current frames
        ssim = compare_ssim(distorted_frame_gray, good_frame_gray)

        # Append the SSIM value to the list
        ssim_values.append(ssim)

        # Check if distortion happens (e.g., SSIM is below a threshold)
        if ssim < 0.75:
            distorted_frame_numbers.append(len(ssim_values))

        # Get the timestamp of the current frame and append to the list
        current_frame_time = distorted_video.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamps.append(current_frame_time)

    # Release the videos
    distorted_video.release()
    good_video.release()

    return ssim_values, distorted_frame_numbers, frame_timestamps

# Rest of the code remains the same

# Example usage
distorted_video_path = r"C:/OTT_PROJECT/SSIM and PSNR/output_video_distorted_middle.avi"
good_video_path = r"C:/OTT_PROJECT/SSIM and PSNR/oxford_referance.mp4"

# Calculate SSIM values for each frame in the distorted video
ssim_values, distorted_frame_numbers, frame_timestamps = calculate_ssim_for_each_frame(distorted_video_path, good_video_path)

# Create a list of frame numbers for x-axis
frame_numbers = list(range(1, len(ssim_values) + 1))

# Plot the SSIM values in a line graph
plt.plot(frame_numbers, ssim_values, marker='o', linestyle='-')
plt.xlabel('Frame Number')
plt.ylabel('SSIM Value')
plt.title('SSIM Values for Distorted Video')
plt.grid(True)
plt.show()

# Print SSIM values and frame timestamps for each frame
data = {
    'Frame Number': frame_numbers,
    'SSIM Value': ssim_values,
    'Timestamp (ms)': frame_timestamps
}

df = pd.DataFrame(data)
print(df)

# Save SSIM values, frame numbers, and timestamps to an Excel file
output_excel_path = r"C:\OTT_PROJECT\SSIM and PSNR\SSIM\ssim_values_and_timestamps.xlsx"
df.to_excel(output_excel_path, index=False)
