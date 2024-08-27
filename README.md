# MedDataML

This repository contains code and resources for machine learning models applied to medical data, including both tabular and image data. The project focuses on classification tasks, optimizing model performance for different medical datasets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Contributing](#contributing)


## Introduction

Medical data, both tabular and image-based, often present unique challenges for classification tasks. This repository provides a comprehensive approach to tackling these challenges using advanced machine learning techniques.

## Features

- **Support for Tabular and Image Data:** Code for handling both types of data.
- **Data Augmentation:** Techniques such as ADASYN for balancing tabular datasets.
- **Custom CNN Model:** A custom `ParallelCNN` model for image classification.
- **Focal Loss Function:** Used for handling class imbalance in the datasets.
- **Optuna for Hyperparameter Tuning:** Leveraging Optuna to optimize model hyperparameters for better performance.
- **Cross-Validation:** Stratified K-Fold cross-validation to ensure robust model evaluation.

## Installation

To use this repository, clone it to your local machine:

```bash
git clone https://github.com/your-username/MedDataML.git
cd MedDataML
```
Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- tabular/: Code and resources for tabular medical data classification.
- image/: Code and resources for medical image classification.
- models/: Custom models and architectures used in the project.
- data/: Placeholder directory for datasets (not included in the repo).
- results/: Directory to store results from experiments.

## Model Training and Evaluation
The models are trained using the train_model function, which includes both training and validation processes. The evaluation is done using a custom test_model function that provides detailed classification reports.

## Hyperparameter Optimization
Optuna is used for hyperparameter optimization. The `objective` function in the image folder's `main.ipynb` is designed to maximize the F1 scores for different class labels.

## Results 

- Results from the cyclometry flow data csv file --> ![image](https://github.com/user-attachments/assets/f807517a-e8fd-4c9e-8337-f4024eff3ee0)
- Results after applying linear transformation and data augmentation to the medical image data supplied -->
![image](https://github.com/user-attachments/assets/0256c7a6-840d-4799-a9a9-0c2dfc04edb0)
  
## Contributing
Contributions to this project are welcome! If you have ideas for improvements or new features, feel free to fork the repository and submit a pull request. Please ensure that your code adheres to the existing coding style and is well-documented.



