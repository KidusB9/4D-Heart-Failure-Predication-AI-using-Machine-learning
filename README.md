# 4D-Heart-Failure-Predication-AI-using-Machine-learning-algorthims

![photo_2023-04-13_01-05-35](https://user-images.githubusercontent.com/107410165/231668415-ad1dd7bb-e1f2-4234-a780-c873b3bc2943.jpg)
![photo_2023-04-13_01-05-41](https://user-images.githubusercontent.com/107410165/231668441-62c895f2-80b7-4eee-8464-d54bf33fa728.jpg)


# 4D-Heart-Failure-Predication-AI-using-Machine-learning-algorithms

![Project Image](photo_2023-04-13_01-05-35.jpg)

## Table of Contents

1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Technologies](#technologies)
4. [Code Overview](#code-overview)
5. [Setup](#setup)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

The concept of the 4D heart refers to a comprehensive, time-sequential representation of a beating heart, showcasing its progression from diastole to systole. This is achieved by creating a series of 3D models that accurately depict the heart's internal structures throughout its rhythmic cycle. Although modern medical imaging technologies, such as computed tomography (CT) and magnetic resonance imaging (MRI), have made it possible to generate detailed 3D data blocks during the cardiac cycle, visualizing the intricate intracardiac processes of a beating heart in 3D still requires segmenting the heart at each phase. This becomes particularly important in cases of dynamic cardiac pathologies, like hypertrophic obstructive cardiomyopathy, where a meticulously segmented 4D heart model can be instrumental in devising effective surgical strategies.

Currently, the process of segmenting these images is a laborious, manual task that involves carefully distinguishing the heart from other structures in the scan. Our objective is to streamline this process by automating myocardial wall segmentation throughout the entire cardiac cycle. To achieve this, we plan to create a machine-learning model capable of identifying the heart and generating accurate masks for the given CT images.

Once the segmentation is complete, we will utilize the machine-learning model's output to construct a 4D heart model that captures the heart's dynamic nature. By developing and refining these innovative heart modeling techniques, we aim to propel medical imaging technology to new heights, thereby enhancing diagnostic and surgical planning capabilities in the field of cardiology.

## Objective

The primary objective is to automate myocardial wall segmentation throughout the cardiac cycle using advanced machine learning algorithms and techniques. This automation will facilitate the construction of a 4D heart model, providing a new level of insight into cardiac activities for improved surgical strategies.

## Technologies

- Python 3.x
- Jupyter Notebook
- Quantum Machine Learning Algorithms
- Coronary Computed Tomography Angiography (CCTA)
- Libraries: Numpy, Pydicom, SimpleITK, qiskit, TensorFlow

## Code Overview

This repository contains specialized Python code and Jupyter Notebooks focused on heart failure prediction using machine learning algorithms. Below is a detailed description of each key file:

### Quantum Machine Learning.py
This Python file is geared towards implementing quantum machine learning algorithms for predicting heart failure.

#### Key Sections:
- Importing Libraries: Libraries such as numpy, pandas, and qiskit are imported.
- Data Preprocessing: Functions for data preprocessing are defined.
- Quantum Circuit Creation: Functions to create quantum circuits are defined.
- Model Training: Code for training the quantum machine learning model is included.
- Evaluation: Code for evaluating the model's performance is present.

### cardiacsegmentationccta.py
This Python file is specialized for cardiac segmentation using CCTA images.

#### Key Sections:
- Importing Libraries: Libraries such as numpy, pydicom, and SimpleITK are imported.
- Reading DICOM Files: Functions for reading DICOM files are defined.
- Image Preprocessing: Functions for preprocessing CCTA images are included.
- Segmentation: Code for cardiac segmentation is present.

### left-atrial-segmentation-all-process-nii-doc (2).ipynb
This Jupyter Notebook is used for segmenting the left atrial region of the heart, possibly using NIfTI (.nii) files.

## Setup

1. Clone this repository.
    ```bash
    git clone https://github.com/Kidus-berhanu/4D-Heart-Failure-Predication-AI-using-Machine-learning-algorithms.git
    ```

2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3. Run Jupyter Notebook.
    ```bash
    jupyter notebook
    ```

## Usage

1. Open `Quantum Machine Learning.py` or `cardiacsegmentationccta.py` in Jupyter Notebook.
2. Run all cells to see the segmentation results and 4D model generation.

## Contributing

Feel free to contribute to this project. Fork the project, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
