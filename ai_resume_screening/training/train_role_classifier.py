import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "normalized_classification.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_models():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data not found: {DATA_PATH}. Run prepare_data.py first.")
        return
        
    df = pd.read_csv(DATA_PATH).dropna()
    X_raw = df['resume_text']
    y_raw = df['resume_role'].astype(str)
    
    # Stratified filtering for rare classes if any, keep simple:
    val_counts = y_raw.value_counts()
    valid_classes = val_counts[val_counts >= 5].index
    
    df = df[df['resume_role'].isin(valid_classes)]
    X = df['resume_text']
    y = df['resume_role']
    
    logger.info(f"Training Role Classifier on {len(X)} records across {len(valid_classes)} classes...")
    
    # 1. Feature Extraction pipeline
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info(f"Vectorization complete. Feature shape: {X_train_vec.shape}")
    
    # 2. Train and Evaluate Models
    # Model A: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_vec, y_train)
    
    lr_pred = lr.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    # Model B: Linear SVM
    svm = LinearSVC(max_iter=2000, class_weight='balanced', random_state=42)
    svm.fit(X_train_vec, y_train)
    
    svm_pred = svm.predict(X_test_vec)
    svm_acc = accuracy_score(y_test, svm_pred)
    
    logger.info(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    logger.info(f"Linear SVM Accuracy: {svm_acc:.4f}")
    
    best_model = svm if svm_acc > lr_acc else lr
    best_name = "Linear SVM" if svm_acc > lr_acc else "Logistic Regression"
    
    logger.info(f"Best Model Strategy Chosen: {best_name}")
    
    # Evaluate best formally
    print(f"\n--- {best_name} Classification Report ---")
    print(classification_report(y_test, best_model.predict(X_test_vec), zero_division=0))
    
    # Save the winner
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "role_classifier.pkl"))
    logger.info("Models persisted to disk.")

if __name__ == "__main__":
    train_models()
