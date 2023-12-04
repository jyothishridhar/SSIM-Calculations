import cv2
import streamlit as st
import numpy as np
import pandas as pd
import requests
import base64
from io import BytesIO
from skimage.metrics import structural_similarity as compare_ssim
import sys
print(sys.path)

def download_video(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return file_name

def calculate_ssim(frame1, frame2):
    ssim = compare_ssim(frame1, frame2)
    return ssim

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
        ssim = calculate_ssim(distorted_frame_gray, good_frame_gray)

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

# Streamlit app code
st.title("SSIM Calculation Demo")

# URLs for the distorted and reference videos
distorted_video_url = "https://github.com/jyothishridhar/SSIM-Calculations/raw/main/distorted.avi"
good_video_url = "https://github.com/jyothishridhar/SSIM-Calculations/raw/main/referance.mp4"

# Download videos
distorted_video_path = download_video(distorted_video_url, 'distorted.mp4')
good_video_path = download_video(good_video_url, 'reference.mp4')

# Add download links
st.markdown(f"**Download Distorted Video**")
st.markdown(f"[Click here to download the Distorted Video]({distorted_video_url})")

st.markdown(f"**Download Reference Video**")
st.markdown(f"[Click here to download the Reference Video]({good_video_url})")

# Add button to run SSIM calculation
if st.button("Run SSIM Calculation"):
    # Calculate SSIM values for each frame in the distorted video
    ssim_values, distorted_frame_numbers, frame_timestamps = calculate_ssim_for_each_frame(distorted_video_path, good_video_path)

    # Create a list of frame numbers for x-axis
    frame_numbers = list(range(1, len(ssim_values) + 1))

    # Plot the SSIM values in a line chart using Streamlit
    st.line_chart(pd.DataFrame({"Frame Number": frame_numbers, "SSIM Value": ssim_values}).set_index("Frame Number"))

    # Display the result on the app
    st.success("SSIM calculation completed!")

    # Display the SSIM values and frame timestamps
    data = {
        'Frame Number': frame_numbers,
        'SSIM Value': ssim_values,
        'Timestamp (ms)': frame_timestamps
    }

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Save SSIM values, frame numbers, and timestamps to an Excel file
    st.markdown(get_excel_link(df, "Download SSIM Report"), unsafe_allow_html=True)
