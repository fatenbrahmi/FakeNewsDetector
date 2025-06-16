# src/predict.py
import joblib #Charger les modèles ML pré-entraînés
import pandas as pd
from pathlib import Path
import sys
import json

# Fix import path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import predict_news, analyze_text_components

# Configure paths
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

def load_artifacts():# Charger les trois composants essentiels du modèle:modele+vectoriser+metadata
    """Load model artifacts with error handling"""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
            
        return model, vectorizer, metadata
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        sys.exit(1)

def analyze_single_text(text, model, vectorizer): # Analyser un seul texte saisi par l'utilisateur.
    """Analyze a single text with detailed output"""
    result = predict_news(text, model, vectorizer)
    
    print("\n Analysis Results:")
    print(f"Prediction: {result['label']}") #La prédiction (Fake News ou Real News).
    print(f"Confidence: {result['confidence']:.1%}") # La confiance de la prédiction (en pourcentage).
    print("\nProbability Breakdown:") #Les probabilités pour chaque classe.
    print(f"Fake News: {result['raw_proba']['Fake News']:.1%}")
    print(f"Real News: {result['raw_proba']['Real News']:.1%}")
    
    # Show influential features
    analysis = analyze_text_components(text, vectorizer)
    print("\nTop Influential Features:")
    for feat in analysis['top_features']:
        print(f"- {feat['feature']} (weight: {feat['weight']:.4f})")
    
    return result

def analyze_batch(input_path, output_path, model, vectorizer):
    """Process a CSV file containing multiple texts"""
    try:
        df = pd.read_csv(input_path)
        if 'text' not in df.columns:
            raise ValueError("CSV must contain 'text' column")
            
        print(f"\nProcessing {len(df)} articles...")
        results = []
        
        for i, row in df.iterrows():
            res = predict_news(row['text'], model, vectorizer)
            results.append({
                'text_id': i,
                'text': row['text'][:100] + "...",  # Preview
                'prediction': res['label'],
                'confidence': res['confidence'],
                'fake_prob': res['raw_proba']['Fake News'],
                'real_prob': res['raw_proba']['Real News']
            })
            
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return result_df
        
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        sys.exit(1)

def main():
    # Load model artifacts
    model, vectorizer, metadata = load_artifacts()
    
    print("\n" + "="*50)
    print(f"Fake News Detector (v{metadata['version']})")
    print(f"Model Type: {metadata['model_type']}")
    print(f"Trained: {metadata['training_date']}")
    print(f"Validation Accuracy: {metadata['metrics']['validation_accuracy']:.1%}")
    print("="*50 + "\n")
    
    while True:
        print("\nOptions:")
        print("1. Analyze single text")
        print("2. Process CSV file")
        print("3. Exit")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            text = input("\nEnter the news text to analyze:\n> ")
            analyze_single_text(text, model, vectorizer)
        elif choice == "2":
            input_csv = input("Enter input CSV path: ").strip()
            output_csv = input("Enter output CSV path: ").strip()
            analyze_batch(input_csv, output_csv, model, vectorizer)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()