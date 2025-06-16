
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import joblib

def load_data():
    """Load and validate all datasets with strict type checking"""
    try:
        # Load datasets
        train_df = pd.read_csv("data/processed/train_cleaned.csv")
        val_df = pd.read_csv("data/processed/val_cleaned.csv")
        test_df = pd.read_csv("data/processed/test_cleaned.csv")
        
        # Validate data
        for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            print(f"\n{name} Set:")
            print(f"Total samples: {len(df)}")
            print(f"Fake news: {sum(df['label']==0)} ({sum(df['label']==0)/len(df):.1%})")
            print(f"Real news: {sum(df['label']==1)} ({sum(df['label']==1)/len(df):.1%})")
            
            # Clean and validate
            df.dropna(subset=['cleaned_text', 'label'], inplace=True)
            df['label'] = df['label'].astype(int)
            invalid_labels = df[~df['label'].isin([0, 1])]
            if not invalid_labels.empty:
                print(f"Warning: {len(invalid_labels)} invalid labels removed")
                df = df[df['label'].isin([0, 1])]
        
        return {
            'X_train': train_df['cleaned_text'],
            'y_train': train_df['label'].values,
            'X_val': val_df['cleaned_text'],
            'y_val': val_df['label'].values,
            'X_test': test_df['cleaned_text'],
            'y_test': test_df['label'].values
        }
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        raise

def plot_evaluation(y_true, y_pred, y_proba, dataset_name=""):
    """Generate comprehensive evaluation plots"""
    # Confusion Matrix
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision, marker='.')
    plt.title(f'Precision-Recall Curve ({dataset_name})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.tight_layout()
    plt.savefig(f'models/evaluation_{dataset_name.lower()}.png')
    plt.close()

def train_model(X_train, y_train, X_val, y_val):
    """Train optimized and calibrated classifier"""
    print("\nTraining model...")
    
    # Initialize vectorizer with enhanced parameters
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),  # Include trigrams
        stop_words='english',
        min_df=5,            # Ignore rare terms
        max_df=0.7,          # Ignore overly common terms
        token_pattern=r'(?u)\b[A-Za-z]{2,}\b'  # Only words with 2+ chars
    )
    
    # Vectorize text
    print("Vectorizing text data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Calculate class weights (inverse of class frequencies)
    class_counts = Counter(y_train)
    class_weights = {
        0: 1 / class_counts[0],
        1: 1 / class_counts[1]
    }
    
    # Initialize and train model
    base_model = LogisticRegression(
        class_weight=class_weights,
        penalty='l2',
        C=0.5,  # Regularization strength
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    
    # Calibrate for better probabilities
    model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    model.fit(X_train_vec, y_train)
    
    # Cross-validation
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(
        model, X_train_vec, y_train, 
        cv=5, scoring='accuracy'
    )
    print(f"Mean Accuracy: {cv_scores.mean():.2%}")
    print(f"Std Dev: {cv_scores.std():.2%}")
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X, y, dataset_name=""):
    """Generate comprehensive evaluation metrics"""
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)
    y_proba = model.predict_proba(X_vec)[:, 1]  # Probability of class 1 (Real)
    
    print(f"\n{dataset_name} Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y, y_pred):.2%}")
    print(f"AUC-ROC: {roc_auc_score(y, y_proba):.2%}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Fake', 'Real']))
    
    # Plot evaluation metrics
    plot_evaluation(y, y_pred, y_proba, dataset_name)
    
    return y_pred, y_proba

def save_artifacts(model, vectorizer, metrics):
    """Save all model artifacts with metadata"""
    os.makedirs("models", exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, 'models/fake_news_model.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    # Save metadata
    metadata = {
        'model_type': 'CalibratedLogisticRegression',
        'classes': ['Fake', 'Real'],
        'version': '2.0',
        'training_date': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nModel artifacts saved to /models directory")

def main():
    print("Starting Fake News Classifier Training")
    
    # Load and validate data
    data = load_data()
    
    # Train model
    model, vectorizer = train_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Evaluate on all sets
    metrics = {}
    for name, X, y in [
        ('Training', data['X_train'], data['y_train']),
        ('Validation', data['X_val'], data['y_val']),
        ('Test', data['X_test'], data['y_test'])
    ]:
        y_pred, y_proba = evaluate_model(model, vectorizer, X, y, name)
        metrics[name.lower() + '_accuracy'] = accuracy_score(y, y_pred)
        metrics[name.lower() + '_auc_roc'] = roc_auc_score(y, y_proba)
    
    # Save everything
    save_artifacts(model, vectorizer, metrics)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()