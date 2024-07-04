from ultralytics import YOLO
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import time
import os

model_path = 'best.onnx'

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(model_path)
        return model

    def predict(self, frame):
        res = self.model(frame, conf=float(30/100))
        return res

    def plot_boxes(self, res):
        xyxys = []
        conf = []
        class_id = []

        for r in res:
            boxes = r.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            conf.append(boxes.conf)
            class_id.append(boxes.cls)

        return res[0].plot(), xyxys, conf, class_id

    def __call__(self, source):
        start = st.sidebar.button("Detect uploaded images")
        if start:
            # show_progress()
            columns = st.columns(2)
            idx = 0
            for s in source:
                col = columns[idx % 2]
                with col:
                    image = Image.open(s)
                    res = self.predict(image)
                    res_plotted, xyxys, conf, class_id = self.plot_boxes(res)
                    res_plotted = cv2.resize(res_plotted, (360,640))
                    st.text("Location: Not identify")
                    # st.write(xyxys)
                    # st.write(conf)
                    # st.write(class_id)
                    st.image(res_plotted)
                idx += 1

def show_progress():
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

st.set_page_config(
    page_title="demo",
    page_icon="🤖", 
    initial_sidebar_state="expanded"
)
st.title(":green[BOX DETECTION WITH AI 🤖]")
st.write("---")
img_list = []
options = st.sidebar.selectbox("Choose which data would be use", ("Sample data", "Custom data"))
if options == "Sample data":
    img_list = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]

elif options == "Custom data":
    img_list = st.sidebar.file_uploader("Upload your images", type=["png", "jpg"], accept_multiple_files=True)

if len(img_list) == 0:
    st.sidebar.warning("AI is not ready for detecting, please upload at least one image", icon="⚠️")
else:
    st.sidebar.success("AI is ready for detecting", icon="✅")
    detector = ObjectDetection()
    detector(img_list)