from ultralytics import YOLO
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_authenticator as stauth
from PIL import Image
import cv2

model_path = 'bestv9c50.pt'

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(model_path)
        return model

    def predict(self, frame):
        res = self.model(frame, conf=float(85/100))
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
        start = st.sidebar.button("画像から箱判定開始する")
        if start:
            columns = st.columns(2)
            dataframe = {'画像ファイル': [], '判定結果の箱': [], "判定の確率": [], "備考": []}
            idx = 0

            for s in range(len(source)):
                col = columns[idx % 2]
                image = Image.open(source[s])
                res = self.predict(image)
                res_plotted, conf, class_id = self.plot_boxes(res)
                res_plotted = cv2.resize(res_plotted, (360,640))

                if len(class_id[0]) == 0:
                    dataframe['画像ファイル'].append(source_name[s])
                    dataframe["判定結果の箱"].append(" ")
                    dataframe["判定の確率"].append(" ")
                    dataframe["備考"].append(" 判定不可")
                else:
                    append_value_to_df(class_id, conf, dataframe)
                    for _ in range(len(class_id[0])):
                        dataframe["画像ファイル"].append(source_name[s])

                with col:
                    st.write(source_name[s])
                    st.image(res_plotted)
                idx += 1
            data = pd.DataFrame(dataframe)
            sorted_data = data.sort_values(by="判定の確率", ascending=False).reset_index(drop=True)
            st.table(sorted_data)

def append_value_to_df(class_id, confidence, df):
    for i in range(len(class_id[0])):
        if class_id[0][i] == 0:
            df["判定結果の箱"].append("box_1")
        elif class_id[0][i] == 1:
            df["判定結果の箱"].append("box_2")
        elif class_id[0][i] == 2:
            df["判定結果の箱"].append("box_4")
        elif class_id[0][i] == 3:
            df["判定結果の箱"].append("box_5")
        elif class_id[0][i] == 4:
            df["判定結果の箱"].append("box_7")
        elif class_id[0][i] == 5:
            df["判定結果の箱"].append("box_8")
        df["判定の確率"].append(f"{round((confidence[0][i] * 100), 2)} %")
        df["備考"].append("判定可能")

st.set_page_config(
    page_title="demo",
    page_icon="🤖", 
    initial_sidebar_state="collapsed"
)
st.title(":green[箱をAIで判定する 🤖]")
st.write("---")
img_list = []
img_name = []
options = st.sidebar.selectbox("使う画像の設定", ("サンプルデータを使う", "箱の画像（複）を指定する"))

if options == "サンプルデータを使う":
    img_list = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
    img_name = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
elif options == "箱の画像（複）を指定する":
    img_list = st.sidebar.file_uploader("選択した画像", type=["png", "jpg"], accept_multiple_files=True)
    img_name = [file.name for file in img_list]

if len(img_list) == 0:
    st.sidebar.warning("判定する画像を1枚以上指定してください", icon="⚠️")
else:
    st.sidebar.success("判定開始可能になりました", icon="✅")
    detector = ObjectDetection()
    detector(img_list, img_name)