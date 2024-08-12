from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Image Demo")
st.title("画像ファイルから物体検出")
model = YOLO('yolov8n.pt')
class_names_map = model.names

with st.form("my_form"):
    selected_classes = st.multiselect('検出したいクラスを選択してください', class_names_map.values(), default=['apple', 'person']) 
    uploaded_image = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])
    st.form_submit_button(label='Submit')

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)  # Display the uploaded image
    keys = [key for key, value in class_names_map.items() if value in selected_classes]
    results = model(image, classes=keys)

    annotated_image = results[0].plot()  # OpenCV形式（BGR）で返される
    # StreamlitはPIL形式（RGB）を期待しているため BGRからRGBに変換
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image_pil = Image.fromarray(annotated_image_rgb)
    st.image(annotated_image_pil, caption="Detected Objects.", use_column_width=True)

