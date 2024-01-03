import streamlit as st
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim_metric
from pytube import YouTube
import base64
import imageio
from io import BytesIO

def get_download_link(df, title):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{title}.csv">Download {title}</a>'
    return href

def download_youtube_video(url, file_name):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4', res='360p').first()
    stream.download()
    return file_name

def calculate_ssim_for_each_frame(distorted_video_url, ssim_threshold):
    yt = YouTube(distorted_video_url)
    stream = yt.streams.filter(file_extension='mp4', res='360p').first()

    ssim_values = []
    video_quality_status = []  # 'Good' or 'Distorted' based on SSIM threshold
    distorted_frame_numbers = []
    frame_timestamps = []

    # Open the video using pytube
    yt = YouTube(distorted_video_url)
    stream = yt.streams.filter(file_extension='mp4', res='360p').first()

    # Iterate over frames
    with imageio.get_reader(stream.url, 'ffmpeg') as video_reader:
        for i, frame in enumerate(video_reader):
            # Convert to grayscale
            distorted_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # You may add more preprocessing steps if needed

            # Compare with itself
            ssim = ssim_metric(distorted_frame_gray, distorted_frame_gray)

            ssim_values.append(ssim)

            # Determine Video Quality Status based on SSIM threshold
            if ssim < ssim_threshold:
                video_quality_status.append('Distorted')
                distorted_frame_numbers.append(i + 1)
            else:
                video_quality_status.append('Good')

            current_frame_time = i * (1 / video_reader.get_meta_data()['fps']) * 1000  # Convert frame number to timestamp in milliseconds
            frame_timestamps.append(current_frame_time)

    return ssim_values, video_quality_status, distorted_frame_numbers, frame_timestamps

st.title("SSIM Calculation Demo")

distorted_video_url = "https://www.youtube.com/watch?v=UzYiAq2nAOU"

st.markdown(f"**YouTube Distorted Video URL:** {distorted_video_url}")

# Add SSIM threshold slider
ssim_threshold = st.slider("Select SSIM Threshold", min_value=0.0, max_value=1.0, value=0.6)

if st.button("Run SSIM Calculation"):
    distorted_video_path = download_youtube_video(distorted_video_url, 'distorted.mp4')

    ssim_values, video_quality_status, distorted_frame_numbers, frame_timestamps = calculate_ssim_for_each_frame(
        distorted_video_url, ssim_threshold
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
