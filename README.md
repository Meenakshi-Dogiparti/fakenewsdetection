# Fake News Detection

A machine learning-based system for detecting fake news articles using Natural Language Processing techniques.

## Overview

This project implements a fake news detection system that uses TF-IDF vectorization and Logistic Regression to classify news articles as either "real" or "fake". The code has been corrected to work in any Python environment (not just Google Colab) and includes proper error handling and documentation.

## Issues Fixed

The original code had several issues that have been corrected:

1. **Google Colab Dependency**: Removed `from google.colab import files` and replaced with proper file handling
2. **Missing Error Handling**: Added comprehensive error handling for file operations and model training
3. **Duplicate Code**: Removed duplicate prediction code blocks
4. **Hard-coded Paths**: Made file paths more flexible and added sample data creation
5. **sklearn Warnings**: Fixed UndefinedMetricWarning by using `zero_division=0` parameter
6. **Poor Structure**: Reorganized code into a proper class-based structure
7. **Missing Documentation**: Added comprehensive documentation and type hints

## Files

- `fake_news_detector.py`: Main corrected implementation with full class structure
- `Untitled4.ipynb`: Corrected Jupyter notebook (original with fixes)
- `requirements.txt`: List of required dependencies
- `README.md`: This documentation file

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python fake_news_detector.py
```

## Usage

### Using the Class-based Implementation

```python
from fake_news_detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Load and train with your data
data = detector.load_data('your_data.csv')
X, y = detector.prepare_data(data)
results = detector.train(X, y)

# Make predictions
prediction = detector.predict("Your news article text here")
probabilities = detector.predict_proba("Your news article text here")
```

### Using the Jupyter Notebook

Open `Untitled4.ipynb` in Jupyter and run the cells. The notebook will automatically create sample data if no dataset is available.

## Data Format

The system expects a CSV file with the following columns:
- `text`: The news article text
- `label`: The label ('real' or 'fake')

Example:
```csv
text,label
"Government announces new policies for education sector.",real
"Breaking: Aliens have landed in downtown!",fake
```

## Sample Data

If no dataset is provided, the system will automatically create sample data for testing purposes.

## Model Performance

The system uses:
- TF-IDF vectorization for text feature extraction
- Logistic Regression with balanced class weights
- Stratified train-test split for better evaluation
- Comprehensive error handling and warnings suppression

## Contributing

Feel free to submit issues and enhancement requests!