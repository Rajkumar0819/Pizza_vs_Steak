import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2

model = load_model("pizza_steak_model.h5")

value = "Pizza vs Steak Identification Model".upper()

# Function to resize image to (224, 224)
def resize_image(image):
    resized_image = cv2.resize(image, (224,224))
    resized_image = np.asarray(resized_image)
    resized_image = resized_image / 255.
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image

# Main function
def main():
    st.title(value)

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    model_output = 0
    # Check if image is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process button
        if st.button("Process"):
            # Convert PIL image to OpenCV format
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Resize image to (224, 224)
            resized_image = resize_image(cv2_image)
            resized_image = tf.constant(resized_image)
            model_output = model.predict(resized_image)

            if model_output[0][0] > 0.5:
                st.header("Steak")
            else:
                st.header("Pizza")

if __name__ == "__main__":
    main()

