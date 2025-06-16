# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
from sklearn.utils import shuffle

# Fix import path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import clean_text, tokenizer

def load_and_combine_data(fake_path, true_path):
    """Enhanced data loading with better text processing"""
    try:
        # Load datasets with error handling
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        # Verify required columns exist
        for df in [fake_df, true_df]:
            if 'text' not in df.columns:
                raise ValueError("DataFrame must contain 'text' column")
        
        # Add labels and source markers
        fake_df['label'] = 0
        fake_df['source_type'] = 'fake'
        true_df['label'] = 1
        true_df['source_type'] = 'true'
        
        # Combine and shuffle
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        combined_df = shuffle(combined_df, random_state=42)
        
        # Standard apply() instead of progress_apply()
        combined_df['cleaned_text'] = combined_df['text'].apply(
            lambda x: clean_text(str(x)) if pd.notnull(x) else '')
        
        # Add text length feature
        combined_df['text_length'] = combined_df['cleaned_text'].apply(len)
        
        return combined_df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def preprocess_and_save(fake_path, true_path, output_dir):
    """Enhanced preprocessing with validation split"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = load_and_combine_data(fake_path, true_path)
        
        # Split into train, validation, and test sets
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42,
            stratify=df['label']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )
        
        # Save all splits
        train_df.to_csv(os.path.join(output_dir, "train_cleaned.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val_cleaned.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_cleaned.csv"), index=False)
        
        # Enhanced vectorizer with more features
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            max_features=10000,  # Increased features
            ngram_range=(1, 3),  # Prend en compte les mots seuls, les paires et les triplets de mots
            stop_words='english',#Ignore les mots courants en anglais (comme "the", "and").
            min_df=5,            # Ignore les mots qui apparaissent dans moins de 5 documents
            max_df=0.7           # Ignore les mots qui apparaissent dans plus de 70% des documents.
        )
        
        # Application du TF-IDF
        X_train = vectorizer.fit_transform(train_df['cleaned_text'])
        X_val = vectorizer.transform(val_df['cleaned_text'])
        X_test = vectorizer.transform(test_df['cleaned_text'])
        
        # Save vectorizer and feature names
        os.makedirs("models", exist_ok=True)
        with open(os.path.join("models", "tfidf_vectorizer.pkl"), 'wb') as f:
            pickle.dump(vectorizer, f)
        #Les matrices vectorisées des textes et les étiquettes de chaque ensemble.
        return {
            'X_train': X_train,
            'y_train': train_df['label'],
            'X_val': X_val,
            'y_val': val_df['label'],
            'X_test': X_test,
            'y_test': test_df['label'],
            'vectorizer': vectorizer,
            'feature_names': vectorizer.get_feature_names_out()
        }
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    FAKE_DATA_PATH = os.path.join("data", "raw", "Fake.csv")
    TRUE_DATA_PATH = os.path.join("data", "raw", "True.csv")
    PROCESSED_DIR = os.path.join("data", "processed")
    
    print("Starting enhanced data preprocessing...")
    processed_data = preprocess_and_save(FAKE_DATA_PATH, TRUE_DATA_PATH, PROCESSED_DIR)
    
    print("\nData preprocessing completed!")
    print(f"Total samples: {len(processed_data['y_train']) + len(processed_data['y_val']) + len(processed_data['y_test'])}")
    print(f"Train samples: {len(processed_data['y_train'])} (Fake: {(processed_data['y_train'] == 0).sum()}, True: {(processed_data['y_train'] == 1).sum()})")
    print(f"Validation samples: {len(processed_data['y_val'])} (Fake: {(processed_data['y_val'] == 0).sum()}, True: {(processed_data['y_val'] == 1).sum()})")
    print(f"Test samples: {len(processed_data['y_test'])} (Fake: {(processed_data['y_test'] == 0).sum()}, True: {(processed_data['y_test'] == 1).sum()})")