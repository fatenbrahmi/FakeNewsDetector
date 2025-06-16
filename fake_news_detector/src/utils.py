
# src/utils.py
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import unicodedata
# Vérifie et télécharge 'punkt' si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    

# Initialize NLP resources with error handling
try:
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(stopwords.words('english'))

# Enhanced stop words list with news-specific terms
EXTRA_STOPWORDS = {
    'u', 're', 'us', 'wa', 'ha', 'said', 'would', 'could', 'one',
    'two', 'also', 'get', 'go', 'like', 'make', 'news', 'article',
    'good', 'tell', 'claim', 'baseball', 'people', 'time', 'country',
    'ing', 'wen', 'trump', 'salem'  # From your sample features
}
STOPWORDS.update(EXTRA_STOPWORDS)

def clean_text(text):
    """
    Enhanced text cleaning pipeline:
    1. Normalize unicode
    2. Handle contractions and special cases
    3. Remove unwanted characters
    4. Clean whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Normalize unicode and remove weird characters
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Remove ellipses and special patterns
    text = re.sub(r'\.{3,}', ' ', text)  # Remove ellipses
    text = re.sub(r'\b\w{1,2}\b', ' ', text)  # Remove 1-2 letter words
    
    # Enhanced contraction handling
    contraction_map = {
        r"won't": "will not",
        r"can't": "cannot",
        r"i'm": "i am",
        r"ain't": "is not",
        r"(\w+)'ll": r"\1 will",
        r"(\w+)n't": r"\1 not",
        r"(\w+)'ve": r"\1 have",
        r"(\w+)'s": r"\1 is",
        r"(\w+)'re": r"\1 are",
        r"(\w+)'d": r"\1 would",
        r"\b(\w+)ing\b": r"\1"  # Handle -ing words
    }

    for pattern, replacement in contraction_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove special chars but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
    
    # Clean whitespace and final processing
    text = ' '.join(text.split())
    return text.lower().strip()

def tokenizer(text, advanced_lemmatization=True):
    """
    Optimized tokenizer with:
    - Smart lemmatization
    - Enhanced stopword removal
    - Context-aware filtering
    """
    if not text.strip():
        return []
    
    tokens = word_tokenize(text)
    processed_tokens = []
    
    for token in tokens:
        # Skip short tokens except meaningful ones
        if len(token) <= 2 and token not in {'no', 'up', 'in', 'it'}:
            continue
            
        # Advanced lemmatization
        if advanced_lemmatization:
            token = lemmatizer.lemmatize(
                lemmatizer.lemmatize(token, pos='v'),
                pos='n'
            )
        
        # Strict filtering
        if (token not in STOPWORDS and 
            not token.isnumeric() and 
            len(token) > 2 and
            not any(c.isdigit() for c in token)):
            processed_tokens.append(token)
    
    return processed_tokens

def preprocess_pipeline(text):
    """Complete preprocessing pipeline with validation"""
    try:
        cleaned = clean_text(str(text))
        tokens = tokenizer(cleaned)
        return tokens if tokens else ['']  # Ensure non-empty output
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return ['']

def predict_news(text, model, vectorizer):
    """
    Robust prediction function with enhanced error handling
    Returns both prediction and explanation
    """
    try:
        tokens = preprocess_pipeline(text)
        processed_text = ' '.join(tokens) if tokens else ''
        
        if not processed_text.strip():
            return {
                "label": "Inconclusive",
                "confidence": 0.0,
                "raw_proba": {"Fake News": 0.5, "Real News": 0.5},
                "processed_text": "",
                "error": "Empty text after preprocessing"
            }
        
        X = vectorizer.transform([processed_text])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        return {
            "label": "Real News" if pred == 1 else "Fake News",
            "confidence": float(proba[pred]),
            "raw_proba": {
                "Fake News": float(proba[0]),
                "Real News": float(proba[1])
            },
            "processed_text": processed_text,
            "top_features": analyze_text_components(text, vectorizer)['top_features']
        }
    except Exception as e:
        return {
            "label": "Error",
            "confidence": 0.0,
            "error": str(e)
        }

def analyze_text_components(text, vectorizer, top_n=10):
    """
    Enhanced feature analysis with better filtering
    """
    tokens = preprocess_pipeline(text)
    processed_text = ' '.join(tokens)
    
    try:
        X = vectorizer.transform([processed_text])
        features = vectorizer.get_feature_names_out()
        weights = X.toarray()[0]
        
        # Filter out low-weight features
        features_df = pd.DataFrame({
            'feature': features,
            'weight': weights
        }).query('weight > 0').sort_values('weight', ascending=False)
        
        return {
            "top_features": features_df.head(top_n).to_dict('records'),
            "processed_text": processed_text
        }
    except Exception as e:
        return {
            "top_features": [],
            "error": str(e)
        }

def get_feature_importance(model, vectorizer, top_n=20):
    """Enhanced feature importance analysis"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = model.coef_[0]
        else:
            return []
        
        features = vectorizer.get_feature_names_out()
        return sorted(
            [(feat, float(imp)) for feat, imp in zip(features, importances)],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
    except Exception as e:
        print(f"Feature importance error: {str(e)}")
        return []