# Train the model and save it
# (Assuming you've already trained the model and saved it as 'my_model.h5')

# Create a Streamlit app for classification
import streamlit as st
from PIL import Image
import numpy as np
import keras
# Load the saved model
model = keras.models.load_model('my_model_1.keras')

# Define Streamlit app layout
st.title('Malaria cell Classification')
file = st.file_uploader("Upload an image", type=["jpg", "png"])

# Classification function
def classify_image(image_path):
    img = Image.open(image_path)
    class_labels = ['Parasitized', 'Uninfected']
    img = img.resize((128, 128))  # Resize image to match input size of the model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    print(prediction, round(prediction[np.argmax(prediction)][0]))
    predicted_class = class_labels[round(prediction[np.argmax(prediction)][0])]
    return predicted_class

# Perform classification when image is uploaded
if file is not None:
    st.image(file, caption='Uploaded Image', use_column_width=True)
    prediction = classify_image(file)
    st.write('Predicted Class:', prediction)
