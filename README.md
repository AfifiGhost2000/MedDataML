# MedDataML

This repository contains code and resources for machine learning models applied to medical data, including both tabular and image data. The project focuses on classification tasks, optimizing model performance for different medical datasets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

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
