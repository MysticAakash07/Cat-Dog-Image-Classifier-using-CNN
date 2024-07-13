# Cat-Dog-Image-Classifier-using-CNN

## Project Description

This project implements a convolutional neural network (CNN) for classifying images of cats and dogs. The dataset consists of images labeled as either cats or dogs, sourced from [this Google Drive folder](https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD).

### Project Links
- **Dataset Link:** [Google Drive Dataset](https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD)
- **Project on Hugging Face:** [Cat-Dog Image Classifier using CNN](https://huggingface.co/spaces/Mystic-Aakash/Cat-Dog-Image-Classifier-using-CNN)

## Model Architecture

The CNN architecture used in this project involves several convolutional layers followed by pooling layers for feature extraction, and then fully connected layers for classification. The exact architecture and details can be found in the code repository linked below.

## Implementation

The implementation utilizes TensorFlow and Keras for building and training the CNN model. Key libraries and dependencies used in the project include:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

## Usage

To use the model, follow these steps:

1. Clone the project repository:
   ```bash
   git clone https://github.com/MysticAakash07/Cat-Dog-Image-Classifier-using-CNN.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app to classify images:
   ```bash
   streamlit run app.py
   ```

## Explanation

### Front End

The front end of the application is built using Streamlit, a Python library that allows the creation of interactive web applications for machine learning and data science projects. The Streamlit app provides an interface for users to upload images and get predictions on whether the uploaded image is of a cat or a dog.

### Model Training

The CNN model is built and trained using TensorFlow and Keras. The key steps involved in model training include:

1. **Loading the Dataset:** The dataset is loaded and preprocessed to be used for training and testing. Images are resized and normalized to ensure uniformity.

2. **Model Architecture:** The model consists of convolutional layers for feature extraction, followed by max pooling layers to reduce the spatial dimensions. Finally, fully connected layers are used for classification.

3. **Compilation and Training:** The model is compiled with the binary cross-entropy loss function and the Adam optimizer. It is then trained on the training dataset for a specified number of epochs.

4. **Evaluation:** The trained model is evaluated on the test dataset to check its performance.

5. **Saving the Model:** The trained model is saved to a file (`cat_dog_model.h5`) for later use in the Streamlit app.

### Making Predictions

In the Streamlit app, the user uploads an image, which is then preprocessed (resized and normalized) and passed to the trained CNN model. The model predicts whether the image is of a cat or a dog, and the result is displayed on the web interface.
