import streamlit as st

st.set_page_config(
    page_title="YOLO object detection use Streamlit",
)


st.write("# Welcome to Demo YOLO object detection useing Streamlit! 👋")

st.markdown(
    """
    このデモではYOLO8を使用して画像または動画からオブジェクトを検出を行います。  
    サイドバーからデモを選択してください。
    """
)

st.sidebar.success("Select a demo above.")
