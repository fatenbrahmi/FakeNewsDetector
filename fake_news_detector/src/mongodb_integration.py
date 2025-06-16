# src/mongodb_integration.py
from pymongo import MongoClient
import joblib
from datetime import datetime
from src.utils import clean_text, tokenizer
from bson import ObjectId
from typing import Dict, List, Union
import sys
import os
import json
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "Fake_News_Detector"
COLLECTION_NAME = "predictions"

class MongoDBHandler:
    def __init__(self):
        """Initialize MongoDB connection and load ML models"""
        try:
            # Initialize MongoDB connection
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Test connection
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]

            # Define model paths
            model_path = BASE_DIR / "models" / "fake_news_model.joblib"
            vectorizer_path = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
            metadata_path = BASE_DIR / "models" / "model_metadata.json"

            # Verify files exist
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

            # Load ML models
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load metadata
            with open(metadata_path) as f:
                self.metadata = json.load(f)

            # Verify loaded models
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model is missing predict_proba method")
            if not hasattr(self.vectorizer, 'transform'):
                raise ValueError("Vectorizer is missing transform method")

        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise

    def predict_news(self, text: str) -> Dict[str, Union[str, float, datetime]]:
        """Make prediction on news text"""
        try:
            if not text.strip():
                raise ValueError("Empty input text")

            # Clean and tokenize text
            cleaned_text = clean_text(text)
            tokens = tokenizer(text)
            
            if not tokens or len(' '.join(tokens)) < 10:
                raise ValueError("Text too short after processing")

            # Vectorize and predict
            X = self.vectorizer.transform([' '.join(tokens)])
            proba = self.model.predict_proba(X)[0]
            prediction = int(proba[1] >= self.metadata['threshold'])
            
            return {
                "label": self.metadata['classes'][prediction],
                "confidence": float(proba[prediction]),
                "text": text,
                "cleaned_text": ' '.join(tokens),
                "timestamp": datetime.now(),
                "model_version": self.metadata['version'],
                "raw_proba": {
                    "Fake News": float(proba[0]),
                    "Real News": float(proba[1])
                }
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    def save_prediction(self, prediction: Dict) -> ObjectId:
        """Save prediction to MongoDB"""
        try:
            # Add validation for required fields
            required_fields = ['label', 'confidence', 'text', 'timestamp']
            if not all(field in prediction for field in required_fields):
                raise ValueError(f"Prediction missing required fields: {required_fields}")
            
            result = self.collection.insert_one(prediction)
            print(f"Prediction saved with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"Failed to save prediction: {str(e)}")
            raise

    def get_recent_predictions(self, limit: int = 5) -> List[Dict]:
        """Retrieve recent predictions from database"""
        try:
            predictions = list(self.collection.find()
                             .sort("timestamp", -1)
                             .limit(limit))
            
            # Convert ObjectId to string for serialization
            for pred in predictions:
                pred['_id'] = str(pred['_id'])
            
            return predictions
        except Exception as e:
            print(f"Failed to fetch predictions: {str(e)}")
            raise

def initialize_database():
    """Initialize database with required collection and validation"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        
        # Create collection with validation if it doesn't exist
        if COLLECTION_NAME not in db.list_collection_names():
            db.create_collection(
                COLLECTION_NAME,
                validator={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['text', 'label', 'confidence', 'timestamp'],
                        'properties': {
                            'text': {'bsonType': 'string'},
                            'label': {'bsonType': 'string'},
                            'confidence': {'bsonType': 'double'},
                            'timestamp': {'bsonType': 'date'},
                            'cleaned_text': {'bsonType': 'string'},
                            'model_version': {'bsonType': 'string'},
                            'raw_proba': {
                                'bsonType': 'object',
                                'properties': {
                                    'Fake News': {'bsonType': 'double'},
                                    'Real News': {'bsonType': 'double'}
                                }
                            }
                        }
                    }
                }
            )
            print(f"Created new collection '{COLLECTION_NAME}' with schema validation")
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting MongoDB integration...")
    try:
        # Initialize database
        initialize_database()
        
        # Create handler instance
        db_handler = MongoDBHandler()

        # Test prediction
        test_text = "The government is hiding secrets about UFOs"
        print(f"\nTesting prediction for: '{test_text}'")

        # Make and save prediction
        prediction = db_handler.predict_news(test_text)
        pred_id = db_handler.save_prediction(prediction)

        print(f"Prediction: {prediction['label']} (confidence: {prediction['confidence']:.2%})")
        print(f"Saved to MongoDB with ID: {pred_id}")

        # Show recent predictions
        print("\nRecent predictions in database:")
        recent_preds = db_handler.get_recent_predictions()
        for pred in recent_preds:
            print(f"- {pred['text'][:50]}... => {pred['label']} ({pred['timestamp']})")

    except Exception as e:
        print(f"Application error: {str(e)}")