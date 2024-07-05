from ultralytics import YOLO
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import cv2

model_path = 'best30v9c.pt'

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(model_path)
        return model

    def predict(self, frame):
        res = self.model(frame, conf=float(80/100))
        return res

    def plot_boxes(self, res):
        conf = []
        class_id = []

        for r in res:
            boxes = r.boxes.cpu().numpy()
            conf.append(boxes.conf)
            class_id.append(boxes.cls)

        return res[0].plot(), conf, class_id

    def __call__(self, source, source_name):
        start = st.sidebar.button("Detect uploaded images")
        if start:
            columns = st.columns(2)
            dataframe = {'image_name': [], 'box_name': [], "confidence": [], "description": []}
            idx = 0

            for s in range(len(source)):
                col = columns[idx % 2]
                image = Image.open(source[s])
                res = self.predict(image)
                res_plotted, conf, class_id = self.plot_boxes(res)
                res_plotted = cv2.resize(res_plotted, (360,640))

                if len(class_id[0]) == 0:
                    dataframe['image_name'].append(source_name[s])
                    dataframe["box_name"].append(" ")
                    dataframe["confidence"].append(" ")
                    dataframe["description"].append("Sorry, I cannot detect any boxes")
                else:
                    append_value_to_df(class_id, conf, dataframe)
                    for _ in range(len(class_id[0])):
                        dataframe["image_name"].append(source_name[s])

                with col:
                    st.write(source_name[s])
                    st.image(res_plotted)
                idx += 1
            data = pd.DataFrame(dataframe)
            sorted_data = data.sort_values(by="confidence", ascending=False).reset_index(drop=True)
            st.table(sorted_data)

def append_value_to_df(class_id, confidence, df):
    for i in range(len(class_id[0])):
        if class_id[0][i] == 0:
            df["box_name"].append("box_1")
        elif class_id[0][i] == 1:
            df["box_name"].append("box_2")
        elif class_id[0][i] == 2:
            df["box_name"].append("box_4")
        elif class_id[0][i] == 3:
            df["box_name"].append("box_5")
        elif class_id[0][i] == 4:
            df["box_name"].append("box_7")
        elif class_id[0][i] == 5:
            df["box_name"].append("box_8")
        df["confidence"].append(f"{round((confidence[0][i] * 100), 2)} %")
        df["description"].append("Detected")

st.set_page_config(
    page_title="demo",
    page_icon="🤖", 
    initial_sidebar_state="expanded"
)
st.title(":green[BOX DETECTION WITH AI 🤖]")
st.write("---")
img_list = []
img_name = []
options = st.sidebar.selectbox("Choose which data would be use", ("Sample data", "Custom data"))

if options == "Sample data":
    img_list = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
    img_name = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
elif options == "Custom data":
    img_list = st.sidebar.file_uploader("Upload your images", type=["png", "jpg"], accept_multiple_files=True)
    img_name = [file.name for file in img_list]

if len(img_list) == 0:
    st.sidebar.warning("AI is not ready for detecting, please upload at least one image", icon="⚠️")
else:
    st.sidebar.success("AI is ready for detecting", icon="✅")
    detector = ObjectDetection()
    detector(img_list, img_name)