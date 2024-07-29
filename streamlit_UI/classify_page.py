import streamlit as st
import sys
sys.path.append('/Users/lap01743/Downloads/WorkSpace/capstone_wed/XMT_Model/')
import image_prediction
from PIL import Image

def app():
    # st.image('./Streamlit_UI/Header.gif', use_column_width=True)
    st.title("Upload a Picture to see if it is a fake (StyleGan) or real face.")
    file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])
    if file_uploaded is not None:
        iamge = image_prediction.process_and_save_image(file_uploaded)
        c1, buff, c2 = st.columns([2, 0.5, 2])
        c2.subheader("Classification Result")
        st.image(iamge,  width=800)



