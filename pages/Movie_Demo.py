import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile


st.set_page_config(page_title="movie Demo")
st.title("動画ファイルから物体検出")

model = YOLO('yolov8n.pt')
class_names_map = model.names

with st.form("my_form"):
    selected_classes = st.multiselect('検出したいクラスを選択してください', class_names_map.values(), default=['person'])
    uploaded_video = st.file_uploader("動画を選択してください", type=["mp4", "mov"])
    st.form_submit_button(label='Submit')

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    # Open the video file
    video_cap = cv2.VideoCapture(temp_file.name)
    stframe = st.empty()
    keys = [key for key, value in class_names_map.items() if value in selected_classes]

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        results = model(frame, classes=keys)
        annotated_frame = results[0].plot()

        # StreamlitはPIL形式（RGB）を期待しているため BGRからRGBに変換
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    video_cap.release()  # Release the video capture object
