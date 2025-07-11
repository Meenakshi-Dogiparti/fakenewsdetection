#!/usr/bin/env python3
"""
Fake News Detection System

This script implements a machine learning model to detect fake news using TF-IDF
vectorization and Logistic Regression. It has been corrected to work in any Python
environment (not just Google Colab).

Author: Corrected version of the original fake news detection code
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
from typing import Optional, Tuple

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class FakeNewsDetector:
    """
    A class for fake news detection using machine learning.
    """
    
    def __init__(self):
        """Initialize the FakeNewsDetector with necessary components."""
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file with proper error handling.
        
        Args:
            file_path (str): Path to the CSV file containing news data
            
        Returns:
            pd.DataFrame: Loaded and cleaned dataset
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file doesn't contain required columns
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
            
            data = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean the data
            data = data.dropna(subset=['text', 'label']).drop_duplicates()
            
            if len(data) == 0:
                raise ValueError("No valid data found after cleaning.")
            
            print(f"Successfully loaded {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training by encoding labels and extracting features.
        
        Args:
            data (pd.DataFrame): Raw dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y)
        """
        try:
            # Encode labels
            data['label_encoded'] = self.label_encoder.fit_transform(data['label'])
            
            # Extract features and labels
            X = data['text'].values
            y = data['label_encoded'].values
            
            print(f"Data prepared: {len(X)} samples with {len(np.unique(y))} unique labels")
            print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """
        Train the fake news detection model.
        
        Args:
            X (np.ndarray): Feature data (text)
            y (np.ndarray): Target labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training results including accuracy and classification report
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Vectorize text data
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report with zero_division parameter to handle warnings
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            
            self.is_trained = True
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'test_size': len(X_test),
                'train_size': len(X_train)
            }
            
            return results
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
    
    def predict(self, news_text: str) -> str:
        """
        Predict whether a news article is fake or real.
        
        Args:
            news_text (str): The news article text to classify
            
        Returns:
            str: The predicted label (original label, not encoded)
            
        Raises:
            RuntimeError: If the model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")
        
        try:
            # Vectorize the input text
            news_vec = self.vectorizer.transform([news_text])
            
            # Make prediction
            prediction = self.model.predict(news_vec)
            
            # Convert back to original label
            label = self.label_encoder.inverse_transform(prediction)
            
            return label[0]
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise
    
    def predict_proba(self, news_text: str) -> dict:
        """
        Get prediction probabilities for a news article.
        
        Args:
            news_text (str): The news article text to classify
            
        Returns:
            dict: Dictionary with class labels and their probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")
        
        try:
            # Vectorize the input text
            news_vec = self.vectorizer.transform([news_text])
            
            # Get prediction probabilities
            proba = self.model.predict_proba(news_vec)[0]
            
            # Map probabilities to original labels
            labels = self.label_encoder.inverse_transform(range(len(proba)))
            
            return dict(zip(labels, proba))
            
        except Exception as e:
            print(f"Error getting prediction probabilities: {str(e)}")
            raise


def create_sample_data(file_path: str = 'sample_news_data.csv') -> None:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        file_path (str): Path where to save the sample data
    """
    sample_data = {
        'text': [
            "Government announces new policies for education sector to improve quality of learning.",
            "Scientists discover breakthrough in renewable energy technology for solar panels.",
            "Local community comes together to support flood victims with donations and volunteer work.",
            "Breaking: Aliens have landed in downtown and are demanding pizza immediately!",
            "Shocking: Local man turns into werewolf every full moon, neighbors confirm sighting.",
            "Miracle cure discovered: Eating chocolate daily prevents all diseases according to fake study.",
            "Celebrity spotted at local grocery store buying normal groceries like a regular person.",
            "New research shows that reading books can improve cognitive function and memory.",
            "Weather forecast predicts sunny skies for the weekend with mild temperatures.",
            "Breaking: Dinosaurs found alive in remote jungle, scientists baffled by discovery."
        ],
        'label': ['real', 'real', 'real', 'fake', 'fake', 'fake', 'real', 'real', 'real', 'fake']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(file_path, index=False)
    print(f"Sample data created: {file_path}")


def main():
    """Main function to demonstrate the fake news detection system."""
    print("Fake News Detection System")
    print("=" * 40)
    
    # Create sample data if no dataset exists
    data_file = 'combined_news_data_cleaned.csv'
    if not os.path.exists(data_file):
        print(f"Dataset '{data_file}' not found. Creating sample data...")
        create_sample_data('sample_news_data.csv')
        data_file = 'sample_news_data.csv'
    
    try:
        # Initialize detector
        detector = FakeNewsDetector()
        
        # Load and prepare data
        data = detector.load_data(data_file)
        X, y = detector.prepare_data(data)
        
        # Train the model
        print("\nTraining the model...")
        results = detector.train(X, y)
        
        # Display results
        print(f"\nTraining completed!")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Training samples: {results['train_size']}")
        print(f"Test samples: {results['test_size']}")
        
        # Print classification report
        print("\nClassification Report:")
        report = results['classification_report']
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"{label}: precision={metrics['precision']:.2f}, "
                      f"recall={metrics['recall']:.2f}, f1-score={metrics['f1-score']:.2f}")
        
        # Test predictions
        print("\nTesting predictions:")
        test_articles = [
            "Government announces new policies for education sector.",
            "Breaking: Aliens have landed in downtown!",
            "Scientists discover breakthrough in renewable energy technology."
        ]
        
        for article in test_articles:
            prediction = detector.predict(article)
            probabilities = detector.predict_proba(article)
            print(f"\nArticle: {article}")
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())