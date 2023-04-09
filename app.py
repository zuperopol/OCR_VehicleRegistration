
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import cv2
import re
try:
    from PIL import Image
except ImportError:
    import Image
import streamlit as st


st.sidebar.write('Select vehicle registration documents image to upload')
img = st.sidebar.file_uploader('',
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)

st.title('Vehicle Registration Documents OCR')

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def img_processing(image): 
  gray = get_grayscale(image)
  thresh = thresholding(gray)
 
  text = pytesseract.image_to_string(thresh, lang = 'tha+eng')
  return [text, thresh]

def prep_data(text):
  license_plate = re.search(r"เลขทะเบียน\s*([\wจัง]+)", text).group(1)
  st.write("License plate:", license_plate)
  province = re.search(r"จังหวัด\s*([^\s]*)", text).group(1)
  st.write("Province:", province)
  brand = re.search(r"ห้อรถ\s*([a-zA-Z]+)", text).group(1)
  st.write("Brand:", brand)
  model = re.search(r"แบบ\s*([a-zA-Z]+)", text).group(1)
  st.write("Model:", model)
  year = re.search(r"ค\.ศ\.\s*(\d+)", text).group(1)
  st.write("Year",year)

  st.success('Images were successfully processed.', icon="✅")

if img is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    tmp=img_processing(opencv_image)
    text= tmp[0]
    text =text.replace(' ','')
    try:
      prep_data(text)
      
    except:
      st.error('Images were not successfully processed. Please try again.', icon="🚨")


