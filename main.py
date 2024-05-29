import streamlit as st
import tensorflow as tf
import numpy as np


# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model_new.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    Dataset ini dibuat ulang menggunakan augmentasi offline dari dataset asli. Dataset asli dapat ditemukan di repositori GitHub ini https://github.com/spMohanty/PlantVillage-Dataset. 
    Dataset ini terdiri dari sekitar 87 ribu gambar RGB dari daun tanaman yang sehat dan yang terkena penyakit, 
    yang dikategorikan ke dalam 38 kelas yang berbeda. Total dataset dibagi menjadi rasio 80/20 untuk set pelatihan dan validasi 
    dengan mempertahankan struktur direktori. Sebuah direktori baru yang berisi 33 gambar uji dibuat kemudian untuk tujuan prediksi.

    #### Content
    1. Train (70295 images)
    2. Test (33 images)
    3. Validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if st.button("Predict") and test_image is not None:
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                       'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                       'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                       'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                       'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                       'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                       'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                       'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                       'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                       'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                       'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                       'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                       'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                       'Tomato___healthy']
        st.success(f"Model is predicting it's a {class_names[result_index]}")
