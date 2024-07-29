
import streamlit_UI.video_prediction as video_prediction
import streamlit as st
import cv2
import tempfile
import shutil

def app():
    st.title("Upload a Video to see if it contains a fake or real face.")
    file_uploaded = st.file_uploader("Choose the Video File", type=["mp4", "avi", "mov"])

    if file_uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            # Write the uploaded file to a temporary file
            shutil.copyfileobj(file_uploaded, tmp_file)
            tmp_file_path = tmp_file.name
            frame_placeholder = st.empty()
            processed_frames = (video_prediction.process_frame(frame) for frame in video_prediction.extract_frames(tmp_file_path))
            for frame in processed_frames:
                frame_placeholder.image(frame, width=800)
                cv2.waitKey(1000 // 60)  
            

