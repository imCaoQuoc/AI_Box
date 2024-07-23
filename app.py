# # External packages
# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import onnx, onnxruntime, torch

# def load_model(model_path):
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = YOLO(model_path)
#     return model

# def milisecond_to_second(milisecond):
#     return float(milisecond/1000)

# def calculate_FPS(milisecond):
#     second = milisecond_to_second(milisecond)
#     return float(1/second)

# def _display_detected_frames(conf, model, st_frame, image):
#     # Predict the objects in the image using the YOLOv9 model
#     res = model.predict(image, conf=conf, stream_buffer=True, agnostic_nms=True)
#     # Plot the detected objects on the video frame
#     res_plotted = res[0].plot()

#     fps = calculate_FPS(res[0].speed["inference"])

#     st_frame.image(res_plotted,
#                    caption=f'Detected Video (FPS: {fps:.2f})',
#                    channels="BGR",
#                    use_column_width=True
#                    )

# def play_stored_video(conf, model):
#     video_source = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
#     status = st.empty()

#     if video_source:
#         st.sidebar.text('Input video')
#         st.sidebar.video(video_source)
#         with open(video_source.name, 'wb') as f:
#             f.write(video_source.read())

#     if video_source is None:
#         status.markdown('<font size= "4"> **Status:** Waiting for input </font>', unsafe_allow_html=True)
#     else:
#         status.markdown('<font size= "4"> **Status:** Ready for detect </font>', unsafe_allow_html=True)

#     if st.sidebar.button('Detect Video Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(video_source.name)
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image
#                                              )
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))

# # Setting page layout
# st.set_page_config(
#     page_title="Object Detection using YOLOv9",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Main page heading
# st.title("Object Detection using YOLOv9")

# confidence = float(30/100)

# model_path = 'best.onnx'
# # model_path = 'best200.pt'

# model = load_model(model_path)
# # play_stored_video(confidence, model)
# play_stored_video(confidence, model)