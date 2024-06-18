Malaria Cell Detection Project
Objective
This project aims to automate the detection of malaria-infected cells using a custom ResNet50 architecture. The model is trained on the Malaria Cell Images Dataset from Kaggle and deployed via a Streamlit web application to provide an accessible diagnostic tool.

Code Development and Execution
We have originally written the code in Jupyter Notebook on the Kaggle website itself to leverage the GPU acceleration it provides for training the model. The notebook includes all the code required to train the model and obtain results.

Running the Code Locally
To run the code locally, you will need to download the dataset. Here is the link for the dataset: Malaria Cell Images Dataset.

Model and Web Application
In the zip file, we have included the model used in the development of the web application, which is done using Streamlit. You can download the model and use it to run the tests and reproduce the results that are provided in the research paper.

Additionally, you can test by uploading a few cell images in the app. The instructions on how to run the app are in the README of the app folder.

Libraries Used
The following libraries are used in the code for both the model and the app:

TensorFlow: For defining and training the deep learning model.
Keras: High-level neural networks API, running on top of TensorFlow.
Pandas: Data manipulation and analysis library.
NumPy: Library for numerical computations.
Matplotlib: Plotting and visualization library.
Pillow: Python Imaging Library for image processing.
Scikit-learn: Machine learning library for evaluation metrics.
Streamlit: For creating the web application.
OS: To interact with the operating system.

To install these libraries, you can use the following command:

pip install tensorflow keras pandas numpy matplotlib pillow scikit-learn streamlit
Make sure you have these libraries installed to ensure the code runs smoothly both for model training and the web application.

Contact Information
For any questions or issues, please contact:

Anurag Singh: as5957@drexel.edu






