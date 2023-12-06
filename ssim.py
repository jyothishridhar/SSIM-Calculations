import cv2
import streamlit as st
import numpy as np
import pandas as pd
import requests
from skimage.metrics import structural_similarity as ssim_metric 
import base64
from io import BytesIO

def get_download_link(df, title):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{title}.csv">Download {title}</a>'
    return href


def download_video(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return file_name

def calculate_ssim(frame1, frame2):
    ssim_value = ssim_metric(frame1, frame2)
    return ssim_value

def calculate_ssim_for_each_frame(distorted_video_path, good_video_path):
    # Open the videos
    distorted_video = cv2.VideoCapture(distorted_video_path)
    good_video = cv2.VideoCapture(good_video_path)

    ssim_values = []
    distorted_frame_numbers = []
    frame_timestamps = []

    while True:
        ret1, distorted_frame = distorted_video.read()
        ret2, good_frame = good_video.read()

        if not ret1 or not ret2:
            break

        distorted_frame_gray = cv2.cvtColor(distorted_frame, cv2.COLOR_BGR2GRAY)
        good_frame_gray = cv2.cvtColor(good_frame, cv2.COLOR_BGR2GRAY)

        ssim = ssim_metric(distorted_frame_gray, good_frame_gray)

        ssim_values.append(ssim)

        if ssim < 0.6:
            distorted_frame_numbers.append(len(ssim_values))

        current_frame_time = distorted_video.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamps.append(current_frame_time)

    distorted_video.release()
    good_video.release()

    return ssim_values, distorted_frame_numbers, frame_timestamps

st.title("SSIM Calculation Demo")

distorted_video_url = "https://github.com/jyothishridhar/SSIM-Calculations/raw/master/distorted.avi"
good_video_url = "https://github.com/jyothishridhar/SSIM-Calculations/raw/master/referance.mp4"

distorted_video_path = download_video(distorted_video_url, 'distorted.mp4')
good_video_path = download_video(good_video_url, 'reference.mp4')

st.markdown(f"**Download Distorted Video**")
st.markdown(f"[Click here to download the Distorted Video]({distorted_video_url})")

st.markdown(f"**Download Reference Video**")
st.markdown(f"[Click here to download the Reference Video]({good_video_url})")

# Add PSNR threshold slider
ssim_threshold = st.slider("Select SSIM Threshold", min_value=0.0, max_value=1, value=0.9)


if st.button("Run SSIM Calculation"):
    ssim_values, distorted_frame_numbers, frame_timestamps = calculate_ssim_for_each_frame(distorted_video_path, good_video_path)

    frame_numbers = list(range(1, len(ssim_values) + 1))

    st.line_chart(pd.DataFrame({"Frame Number": frame_numbers, "SSIM Value": ssim_values}).set_index("Frame Number"))

    st.success("SSIM calculation completed!")

    data = {
        'Frame Number': frame_numbers,
        'SSIM Value': ssim_values,
        'Timestamp (ms)': frame_timestamps
    }

    df = pd.DataFrame(data)

    # Display the dataframe and create a download link
    st.write(df)
    st.markdown(get_download_link(df, "SSIM Report"), unsafe_allow_html=True)
