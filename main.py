import streamlit as st
from keras.models import load_model
from streamlit_option_menu import option_menu
from util import set_background, prediction

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                            ['Brain',
                            'Lung',
                            'Kidney'],
                            menu_icon = 'heart-pulse',
                            icons = ['shield-plus', 'lungs', 'meta'],
                            default_index = 0)

if selected == 'Brain':

    set_background('./bg/brain.png')
    st.header('Glioma/Meningioma/Pituitary Adenoma Prediction')
    st.subheader('Upload a Brain MRI image')
    file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])

    model = load_model('./models/tm224/brain_model.h5', compile = False)

    with open('./models/labels/brain_labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    prediction(file, model, class_names)

if selected == 'Lung':

    set_background('./bg/lung.png')
    st.title('Pneumonia Prediction')
    st.header('Upload a Chest X-Ray image')
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
    model = load_model('./models/tm224/lung_model.h5')

    with open('./models/labels/lung_labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    prediction(file, model, class_names)

if selected == 'Kidney':

    set_background('./bg/kidney.png')
    st.title('Renal Cancer Prediction')
    st.header('Upload a Kidney CT scan image')
    file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])
    model = load_model('./models/tm224/kidney_model.h5', compile = False)

    with open('./models/labels/kidney_labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    prediction(file, model, class_names)

