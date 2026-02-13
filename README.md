# Hydrogel Release Rate Prediction Tool

## Project Overview

The Hydrogel Release Rate Prediction Tool is a Python-based desktop application designed to predict the release rate constant (K h⁻¹) of hydrogels. Utilizing a machine learning model (Random Forest Regressor), the tool models various hydrogel parameters and provides an intuitive graphical user interface for users.
exe file download URL: www.alipan.com/s/su5yudxdq9w

## Key Features

1. **Data Management**
   - Load default dataset or upload custom CSV files
   - Automatic validation of data format and required fields

2. **Model Training**
   - Train models with predefined random forest parameters
   - Automatic categorical variable encoding
   - Real-time training status display

3. **Interactive Prediction**
   - Dynamically generated input forms
   - Real-time hydrogel release rate prediction
   - Clear prediction results display

## Technical Specifications

- **Machine Learning Framework**: scikit-learn
- **GUI Framework**: Tkinter
- **Data Processing**: pandas, numpy
- **Model**: Random Forest Regressor
- **Data Preprocessing**: One-Hot Encoding

## System Requirements

### Software Environment
- Python 3.7 or higher
- Supported OS: Windows, macOS, Linux

### Python Dependencies
See `requirements.txt` file

## Installation Guide

### Method 1: Run from Source Code

1. Clone or download project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python hydrogel_predictor.py
   ```

### Method 2: Package with PyInstaller (Optional)

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Create executable:
   ```bash
   pyinstaller --onefile --windowed hydrogel_predictor.py
   ```


## Usage Instructions

### 1. Launch the Application
After running the program, the interface is divided into three main sections.

### 2. Model Training
- **Step 1**: Click "Upload New Dataset (.csv)" to upload custom data file
  - Or use the default dataset (located at `data/data.csv` in the program directory)
- **Step 2**: Click "Train Model" to start training
- **Status Indicators**:
  - Red: Model not trained
  - Green: Model successfully trained

### 3. Input Feature Values
After successful model training, input fields will be automatically generated:
- Numerical features: Text input boxes
- Categorical features: Dropdown selection boxes

### 4. Execute Prediction
- Click the "Predict" button
- Prediction results will be displayed at the bottom of the interface

## Model Parameter Configuration

The Random Forest Regressor parameters used in the program are as follows:
```python
rf_params = {
    'n_estimators': 102,
    'max_depth': 16,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.3,
    'bootstrap': True,
    'criterion': 'squared_error',
    'random_state': 42,
    'n_jobs': -1
}
```

## Project File Structure
```
hydrogel-predictor/
├── hydrogel_predictor.py    # Main program file
├── requirements.txt         # Dependency list
├── README.md               # Documentation
├── data/                   # Data directory
│   └── data.csv           # Default dataset
└── ...                    # Other resource files
```

## Troubleshooting

### Common Issues and Solutions

1. **"Default dataset not found" warning**
   - Ensure the `data/data.csv` file exists
   - Or use the "Upload New Dataset" button to upload a custom dataset

2. **"The dataset is missing required columns" error**
   - Check if the CSV file contains all required columns
   - Ensure column names are spelled exactly correctly (including spaces and symbols)

3. **Model training failure**
   - Check if data contains non-numeric values
   - Ensure data has no null values or outliers

4. **Inaccurate prediction results**
   - Verify input values are within the reasonable range of training data
   - Consider using more diverse training data

## Customization and Extension

### Modifying Model Parameters
Edit values in the `rf_params` dictionary to adjust Random Forest model parameters.

### Adding New Features
1. Add new feature names to the `numeric_features` or `categorical_features` lists
2. Ensure data files contain corresponding columns
3. The program will automatically adjust the input interface

### Changing Machine Learning Models
Modify the regressor part of the pipeline in the `train_model` method, replacing it with other scikit-learn regression models.

## License
This project is for learning and research purposes only.

## Important Notes
1. This tool is suitable for scientific research and experimental reference; professional judgment should be combined for practical applications
2. Model accuracy depends on the quality and quantity of training data
3. Regular model updates are recommended to adapt to new experimental data
4. Ensure all input values are within reasonable ranges to avoid extreme values causing prediction distortion

## Changelog
- v1.0.0: Initial version release
  - Implemented basic data loading, model training, and prediction functions
  - Provided graphical user interface
  - Supported custom dataset upload

## Technical Support
For questions or suggestions, please check code comments or contact the developer.
