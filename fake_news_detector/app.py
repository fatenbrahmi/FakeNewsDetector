import streamlit as st
from pymongo import MongoClient
import joblib
import json
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils
from src.utils import clean_text, tokenizer

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "Fake_News_Detector"
COLLECTION_NAME = "predictions"

@st.cache_resource
def load_ml_components():
    """Load ML models and components with comprehensive validation and defaults"""
    try:
        model_path = Path("models/fake_news_model.joblib")
        vectorizer_path = Path("models/tfidf_vectorizer.joblib")
        metadata_path = Path("models/model_metadata.json")
        
        # Check files exist
        if not all([model_path.exists(), vectorizer_path.exists(), metadata_path.exists()]):
            missing = [p.name for p in [model_path, vectorizer_path, metadata_path] if not p.exists()]
            raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")
            
        # Load components
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Set defaults for missing metadata
        defaults = {
            'classes': ['Fake News', 'Real News'],
            'threshold': 0.5,
            'version': '1.0',
            'training_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Validate model
        required_model_methods = ['predict', 'predict_proba']
        for method in required_model_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model missing required method: {method}")
                
        # Validate vectorizer
        if not hasattr(vectorizer, 'transform'):
            raise ValueError("Vectorizer missing transform method")
            
        return {
            'model': model,
            'vectorizer': vectorizer,
            'classes': metadata.get('classes', defaults['classes']),
            'threshold': metadata.get('threshold', defaults['threshold']),
            'version': metadata.get('version', defaults['version']),
            'training_date': metadata.get('training_date', defaults['training_date'])
        }
        
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.error("Please ensure:")
        st.error("1. You've run the training script recently")
        st.error("2. All files exist in the models/ directory")
        st.error("3. You're using the same Python environment")
        st.stop()

@st.cache_resource
def init_db():
    """Initialize MongoDB connection with collection validation"""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        db = client[DB_NAME]
        
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
                            'timestamp': {'bsonType': 'date'}
                        }
                    }
                }
            )
        return db[COLLECTION_NAME]
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

# Streamlit App Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 250px;
        font-size: 16px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .correct-prediction {
        background-color: #f0fdf4;
        border-left: 6px solid #10b981;
    }
    .incorrect-prediction {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
    }
    .neutral-prediction {
        background-color: #eff6ff;
        border-left: 6px solid #3b82f6;
    }
    .confidence-meter {
        margin: 1rem 0;
        height: 12px;
        border-radius: 6px;
        background: #e5e7eb;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .analysis-id {
        font-family: monospace;
        font-size: 0.9rem;
        color: #6b7280;
    }
    .stMarkdown h3 {
        margin-top: 1.5rem;
    }
    .feature-importance {
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    .positive-impact {
        color: #10b981;
    }
    .negative-impact {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
ml_components = load_ml_components()
collection = init_db()

def predict_news(text):
    """Enhanced prediction with comprehensive error handling and feature analysis"""
    try:
        if not text.strip():
            raise ValueError("Empty input text")
            
        # Clean and tokenize text using utils.py functions
        cleaned_text = clean_text(text)
        tokens = tokenizer(cleaned_text)
        
        # Verify tokenization produced valid results
        if not tokens or len(' '.join(tokens)) < 10:
            raise ValueError("Text too short or invalid after processing")
        
        # Vectorize and predict
        X = ml_components['vectorizer'].transform([' '.join(tokens)])
        proba = ml_components['model'].predict_proba(X)[0]
        prediction = int(proba[1] >= ml_components['threshold'])
        
        # Get feature importance
        feature_names = ml_components['vectorizer'].get_feature_names_out()
        if hasattr(ml_components['model'], 'feature_importances_'):
            importances = ml_components['model'].feature_importances_
        elif hasattr(ml_components['model'], 'coef_'):
            importances = ml_components['model'].coef_[0]
        else:
            importances = None
        
        top_features = []
        if importances is not None:
            features_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', key=abs, ascending=False).head(10)
            top_features = features_df.to_dict('records')
        
        return {
            "label": ml_components['classes'][prediction],
            "confidence": float(proba[prediction]),
            "text": text,
            "cleaned_text": ' '.join(tokens),
            "timestamp": datetime.now(),
            "model_version": ml_components['version'],
            "raw_proba": {
                "Fake News": float(proba[0]),
                "Real News": float(proba[1])
            },
            "top_features": top_features
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

def save_prediction(prediction):
    """Save prediction with document validation"""
    try:
        result = collection.insert_one({
            k: v for k, v in prediction.items() 
            if k not in ['top_features']  # Don't store features in DB
        })
        return result.inserted_id
    except Exception as e:
        st.error(f"Failed to save prediction: {str(e)}")
        st.stop()

def display_prediction_result(prediction, pred_id):
    """Enhanced result display with interactive elements and feature analysis"""
    # User verifies if prediction is correct
    is_correct = st.radio(
        f"Is this prediction correct?",
        options=["Correct", "Incorrect", "Unsure"],
        key=f"feedback_{pred_id}"
    )
    
    # Determine styling based on user feedback
    if is_correct == "Correct":
        pred_class = "correct-prediction"
        confidence_color = "#10b981"
    elif is_correct == "Incorrect":
        pred_class = "incorrect-prediction"
        confidence_color = "#ef4444"
    else:
        pred_class = "neutral-prediction"
        confidence_color = "#3b82f6"
    
    confidence = prediction['confidence'] * 100
    
    st.markdown(f"""
    <div class="prediction-box {pred_class}">
        <h3 style="margin-top: 0;">{prediction["label"]}</h3>
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence}%; 
                 background: {confidence_color}"></div>
        </div>
        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
        <p class="analysis-id">ID: {str(pred_id)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("View detailed analysis"):
        st.markdown("**Original Text:**")
        st.write(prediction["text"])
        
        st.markdown("**Processed Text:**")
        st.write(prediction["cleaned_text"])
        
        st.markdown("**Probability Breakdown:**")
        proba_df = pd.DataFrame({
            'Class': ['Fake News', 'Real News'],
            'Probability': [f"{prediction['raw_proba']['Fake News']*100:.1f}%", 
                           f"{prediction['raw_proba']['Real News']*100:.1f}%"]
        })
        st.table(proba_df)
        
        if prediction.get('top_features'):
            st.markdown("**Key Influential Features:**")
            for feat in prediction['top_features']:
                impact_class = "positive-impact" if feat['importance'] > 0 else "negative-impact"
                st.markdown(f"""
                <div class="feature-importance">
                    <span class="{impact_class}">{"↑" if feat['importance'] > 0 else "↓"}</span>
                    {feat['feature']} (weight: {feat['importance']:.4f})
                </div>
                """, unsafe_allow_html=True)

def display_recent_predictions(limit=5):
    """Display recent predictions with improved formatting"""
    recent = list(collection.find().sort("timestamp", -1).limit(limit))
    if recent:
        df = pd.DataFrame(recent)[["timestamp", "label", "confidence", "model_version"]]
        df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%d %H:%M')
        df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        st.table(df.rename(columns={
            "timestamp": "Time",
            "label": "Prediction",
            "confidence": "Confidence",
            "model_version": "Version"
        }))
    else:
        st.info("No previous analyses found")

def main():
    st.title("Advanced Fake News Detector")
    st.markdown("""
    Analyze news articles using machine learning to identify potential misinformation.
    The system evaluates text content and provides a confidence score for its assessment.
    """)
    
    # Main content column (removed the second column with examples)
    with st.form("prediction_form"):
        text_input = st.text_area(
            "Paste news article text here:",
            height=300,
            placeholder="Enter the news article content you want to analyze...",
            help="Minimum 50 characters for reliable analysis"
        )
        submitted = st.form_submit_button("Analyze Article", type="primary")
    
    # Handle prediction flow
    if submitted:
        if not text_input.strip():
            st.warning("Please enter some text to analyze")
        elif len(text_input) < 50:
            st.warning("For more accurate results, please enter at least 50 characters")
        else:
            with st.spinner("Analyzing article content..."):
                prediction = predict_news(text_input)
                pred_id = save_prediction(prediction)
                
                st.subheader("Analysis Result")
                display_prediction_result(prediction, pred_id)
                
                st.subheader("Recent Analyses")
                display_recent_predictions()
    
    # Sidebar with detailed information
    with st.sidebar:
        st.title("About This Tool")
        st.markdown("""
        **How It Works:**
        1. Processes text using NLP techniques
        2. Extracts key features using TF-IDF
        3. Classifies using a trained machine learning model
        4. Stores results for reference
        
        **Model Information:**
        - Algorithm: Logistic Regression
        - Version: {version}
        - Decision Threshold: {threshold}
        - Last Trained: {train_date}
        """.format(
            version=ml_components['version'],
            threshold=ml_components['threshold'],
            train_date=ml_components['training_date']
        ))
        
        st.markdown("---")
        st.markdown("**Database Status:**")
        st.write(f"Total predictions: {collection.count_documents({})}")
        
        st.markdown("---")
        st.markdown("""
        **Note:** This tool provides statistical predictions, not absolute truth. 
        Always verify important claims with multiple reliable sources.
        """)

if __name__ == "__main__":
    main()