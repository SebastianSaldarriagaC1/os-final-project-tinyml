# Anomaly Detection with TinyML on ESP32

## Overview

This repository contains the code and resources for deploying a TinyML model for anomaly detection on an ESP32 device. The project aims to showcase the feasibility of using lightweight machine learning models for real-time monitoring applications, particularly in IoT environments.

## System Requirements

To run the application, you will need the following:

- Python environment with Jupyter Notebook or Google Colab for training and preprocessing the model
- Visual Studio Code with the following extensions:
  - PlatformIO extension for building and deploying the project
  - Wokwi extension for simulating the ESP32 device (official documentation on how to install and configure the extension can be found [here](https://docs.wokwi.com/vscode/getting-started))

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/anomaly-detection-tinyml.git
    ```

## Usage

### Notebooks on GitHub

The project includes Jupyter Notebooks for data preprocessing, model training, and evaluation. You can view and execute these notebooks directly on Google Colab by clicking on the "Open on Colab" button in each notebook file. If you wish to make changes, simply make a copy of the notebook to your Google Colab drive.

#### Training the Model

1. Open the Google Colab notebooks (`TinyML01_Data_preprocessing.ipynb`, `TinyML02_Model_training.ipynb` and `TinyML03_Model_validation_and_C_conversion.ipynb`) for training the TinyML model.
2. Follow the instructions in the notebook to preprocess the dataset, train the model, and save the trained model files (`model.tflite` and `temperature_model.h`).

**Note:** If you want to re-clean the dataset in the first Notebook, make sure to change the download link for the rest of them.

### Deploying the Model

1. Open the PlatformIO project (`TemperatureAnomlayDetection_TinyML_ESP32`) in your Visual Studio Code.
2. Ensure you have installed and configured the PlatformIO extension.
3. Build the project to compile the code for the ESP32 device.
4. Install and configure the Wokwi extension for simulating the ESP32 device in Visual Studio Code.
5. Run the "Start Wokwi Simulator" command from the Wokwi extension to launch the simulator with the deployed code.
