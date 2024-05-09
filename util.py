import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names, size):
    image = ImageOps.fit(image, (size, size), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, size, size, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def preprocess_image(image_array, size):
    # Resize the image to the required input size of the model
    resized_image = tf.image.resize(image_array, (size, size))
    # Normalize the image to the range [-1, 1]
    normalized_image = (resized_image / 127.5) - 1
    # Add batch dimension
    preprocessed_image = tf.expand_dims(normalized_image, 0)
    return preprocessed_image

def compute_saliency_map(image, model, size):
    # Convert the PIL image to NumPy array
    image_array = np.array(image)
    # Preprocess the image for the model
    preprocessed_image = preprocess_image(image_array, size)
    # Create a gradient tape to record gradients
    with tf.GradientTape() as tape:
        tape.watch(preprocessed_image)
        # Get the model prediction
        prediction = model(preprocessed_image)
        # Get the top predicted class index
        predicted_class_index = tf.argmax(prediction[0])
        # Get the predicted class score
        predicted_class_score = prediction[0][predicted_class_index]
    # Get the gradients of the predicted class score with respect to the input image
    gradients = tape.gradient(predicted_class_score, preprocessed_image)
    # Compute the saliency map as the absolute gradients
    saliency_map = tf.abs(gradients)
    # Convert the saliency map to a NumPy array
    saliency_map_array = saliency_map.numpy()
    # Reshape the saliency map array to match the image shape
    saliency_map_array = np.mean(saliency_map_array, axis=-1)  # Take mean along the channel axis
    saliency_map_array = np.expand_dims(saliency_map_array, axis=-1)  # Add channel dimension
    saliency_map_array = np.tile(saliency_map_array, (1, 1, 3))  # Tile to match RGB channels
    # Normalize the saliency map array
    saliency_map_array /= np.max(saliency_map_array)
    return saliency_map_array

def save_saliency_map(saliency_map, output_path, cmap='hot'):
    plt.imshow(saliency_map, cmap=cmap, vmin=0, vmax=1)  # Adjust vmin and vmax as needed
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

def prediction(file, model, class_names):
    if file is not None:
        pil_image = Image.open(file).convert('RGB')
        st.image(pil_image, use_column_width=True)
        image = np.array(pil_image)
        class_name, conf_score = classify(pil_image, model, class_names, 224)
        if class_name == 'Healthy':
            st.write("## {}".format(class_name))
        else:
            st.write("## Probability of {}".format(class_name))
        st.write("### Confidence: {}%".format(int(conf_score * 1000) / 10))

        saliency_map = compute_saliency_map(pil_image, model, 224)

        output_folder = 'saliency_maps'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = os.path.join(output_folder, 'saliency_map.png')
        save_saliency_map(saliency_map[0], output_path, cmap='hot')

        st.image(output_path, caption='Saliency Map', use_column_width=True)
