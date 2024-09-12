

# Bone Fracture Classification Project

This project involves building and comparing different deep learning models for classifying bone fractures using X-ray images. The objective is to determine which model best distinguishes between fractured and non-fractured bones.

## Dataset

The dataset for this project is available on [Kaggle: Bone Fracture Dataset](https://www.kaggle.com/datasets/ahmedashrafahmed/bone-fracture). It contains labeled X-ray images of bones, categorized into fractured and healthy.

## Requirements

The following libraries are required:

- TensorFlow
- Keras
- Google Colab (optional)
- Pillow (PIL)
- NumPy
- Matplotlib
- os (for directory management)

Install the dependencies using:
```bash
pip install tensorflow Pillow numpy matplotlib
```

## Setup Instructions

1. Download the dataset from Kaggle and ensure it is organized into appropriate directories.
2. Clone the repository or download the notebook.
3. Install the required dependencies using the above command.
4. Optionally, use Google Colab for running the notebook with Google Drive integration.

## Data Preparation

- **Image Augmentation**: Data augmentation is performed using `ImageDataGenerator`, which includes techniques like random rotations, flips, zooming, and shifts.
- **Preprocessing**: All images are resized and normalized to ensure uniformity before model training.

## Models Used

Several deep learning models are explored and evaluated in this project:

### 1. **VGG16-based Model**
   - Pre-trained VGG16 model used as a feature extractor.
   - Fine-tuning added fully connected layers on top of VGG16 for binary classification.
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy

### 2. **ResNet-based Model**
   - Utilizes the ResNet architecture to handle image classification.
   - ResNet's skip connections are leveraged to improve training performance, especially for deeper layers.
   - Custom fully connected layers added for binary output.
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy

### 3. **Custom CNN Model**
   - A custom-built Convolutional Neural Network (CNN) model designed from scratch.
   - Includes multiple convolutional, pooling, and fully connected layers.
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy

Each model is trained using similar hyperparameters, including batch size, number of epochs, and learning rate.

## Training and Evaluation

- The dataset is split into training, validation, and test sets.
- During training, the models are evaluated based on accuracy, loss, and confusion matrices.
- Hyperparameter tuning is performed to optimize performance across models.

## Results

- The performance of each model is compared based on key metrics: accuracy, loss, precision, recall, and F1-score.
- Final evaluation on the test set provides a comprehensive comparison of the models.
- Results are visualized using plots and confusion matrices for error analysis.

## How to Run

1. Open the notebook in Google Colab or a local environment.
2. Load the dataset using either Google Drive or a local path.
3. Run each model's section to train and evaluate the models on the dataset.
4. Review the results and comparisons in the results section.


