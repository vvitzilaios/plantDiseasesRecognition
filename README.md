# Plant Diseases Identification using Computer Vision

This project utilizes computer vision to identify plant diseases, assisting in early disease detection and crop management.

## Project Overview

Plant diseases can have a significant impact on crop yield and quality. Identifying diseases early 
can help farmers take timely action to prevent the spread of diseases and minimize crop losses. 
This project employs machine learning and computer vision techniques to create a model capable of recognizing 
and classifying plant diseases from images.

## How to Run the Project

To run this project successfully, follow these steps:

### Step 1: Dataset Preparation
#### Dataset used in this project is available at Kaggle's [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

Organize your dataset with the following structure on root directory:

- `data/train/`: Training dataset images organized into subdirectories by disease class.
- `data/valid/`: Validation dataset images organized similarly.

### Step 2: Dependencies

Make sure you have the following Python libraries installed:

- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `tqdm`

You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Training

The following arguments are available:
- --model: Specify the model name (e.g., ResNet9).
- --epochs: Set the number of training epochs.
- --num_samples_per_class: Define the number of samples per class.
- --batch_size: Choose the batch size for training.
- --learning_rate: Set the learning rate.

Train the model by running the main.py script with the following command, adjusting hyperparameters as needed:

```bash
python main.py --model ResNet9 --epochs 20 --num_samples_per_class 500 --batch_size 32 --learning_rate 0.001
```

### Step 4: Evaluation
The trained model will be evaluated on a separate test dataset. 
Test accuracy and various metrics will be computed to assess the model's performance in disease identification.