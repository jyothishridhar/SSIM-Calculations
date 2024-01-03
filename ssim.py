import cv2
import streamlit as st
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim_metric
from pytube import YouTube
import base64
from io import BytesIO

def get_download_link(df, title):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{title}.csv">Download {title}</a>'
    return href

def download_youtube_video(url, file_name):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4', res='360p').first()
    stream.download(file_name)
    return file_name

def calculate_ssim(frame1, frame2):
    ssim_value = ssim_metric(frame1, frame2)
    return ssim_value

def calculate_ssim_for_each_frame(distorted_video_url, good_video_path, ssim_threshold):
    # Open the videos
    good_video = cv2.VideoCapture(good_video_path)

    yt = YouTube(distorted_video_url)
    stream = yt.streams.filter(file_extension='mp4', res='360p').first()
    
    ssim_values = []
    video_quality_status = []  # 'Good' or 'Distorted' based on SSIM threshold
    distorted_frame_numbers = []
    frame_timestamps = []

    cap = cv2.VideoCapture(stream.file_path)

    while True:
        ret1, distorted_frame = cap.read()
        ret2, good_frame = good_video.read()

        if not ret1 or not ret2:
            break

        distorted_frame_gray = cv2.cvtColor(distorted_frame, cv2.COLOR_BGR2GRAY)
        good_frame_gray = cv2.cvtColor(good_frame, cv2.COLOR_BGR2GRAY)

        ssim = ssim_metric(distorted_frame_gray, good_frame_gray)

        ssim_values.append(ssim)

        # Determine Video Quality Status based on SSIM threshold
        if ssim < ssim_threshold:
            video_quality_status.append('Distorted')
            distorted_frame_numbers.append(len(ssim_values))
        else:
            video_quality_status.append('Good')

        current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamps.append(current_frame_time)

    cap.release()
    good_video.release()

    return ssim_values, video_quality_status, distorted_frame_numbers, frame_timestamps

st.title("SSIM Calculation Demo")

distorted_video_url = "https://www.youtube.com/watch?v=UzYiAq2nAOU"
good_video_url = "https://github.com/jyothishridhar/SSIM-Calculations/raw/master/referance.mp4"

good_video_path = download_video(good_video_url, 'reference.mp4')

st.markdown(f"**YouTube Distorted Video URL:** {distorted_video_url}")

# Add SSIM threshold slider
ssim_threshold = st.slider("Select SSIM Threshold", min_value=0.0, max_value=1.0, value=0.6)

if st.button("Run SSIM Calculation"):
    distorted_video_path = download_youtube_video(distorted_video_url, 'distorted.mp4')
    
    ssim_values, video_quality_status, distorted_frame_numbers, frame_timestamps = calculate_ssim_for_each_frame(
        distorted_video_url, good_video_path, ssim_threshold
    )

    frame_numbers = list(range(1, len(ssim_values) + 1))

    # Plot the SSIM values
    st.line_chart(pd.DataFrame({"Frame Number": frame_numbers, "SSIM Value": ssim_values}).set_index("Frame Number"))

    # Plot the Video Quality Status
    st.line_chart(pd.DataFrame({"Frame Number": frame_numbers, "Video Quality Status": video_quality_status}).set_index("Frame Number"))

    st.success("SSIM calculation completed!")

    data = {
        'Frame Number': frame_numbers,
        'SSIM Value': ssim_values,
        'Video Quality Status': video_quality_status,
        'Timestamp (ms)': frame_timestamps
    }

    df = pd.DataFrame(data)

    # Display the dataframe and create a download link
    st.write(df)
    st.markdown(get_download_link(df, "SSIM Report"), unsafe_allow_html=True)
