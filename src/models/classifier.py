"""
Ticket Classification Model

Implements TF-IDF vectorization with Logistic Regression and Random Forest
classifiers for multi-label ticket classification (category and priority).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class TicketClassifier:
    """
    Multi-output classifier for support ticket category and priority prediction.
    
    Architecture:
        - TF-IDF Vectorizer: Convert text to numerical features
        - Logistic Regression: Fast, interpretable classifier (default)
        - Random Forest: Alternative for comparison
    
    Why TF-IDF + Logistic Regression?
        - TF-IDF captures term importance across documents
        - Logistic Regression is fast, interpretable, and performs well on text
        - Easy to deploy and explain to stakeholders
        - Low latency for real-time predictions
    """
    
    # Default model hyperparameters
    DEFAULT_TFIDF_PARAMS = {
        "max_features": 5000,
        "ngram_range": (1, 2),  # Unigrams and bigrams
        "min_df": 2,            # Ignore rare terms
        "max_df": 0.95,         # Ignore very common terms
        "sublinear_tf": True    # Apply logarithmic TF scaling
    }
    
    DEFAULT_LR_PARAMS = {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",  # Handle class imbalance
        "random_state": 42
    }
    
    DEFAULT_RF_PARAMS = {
        "n_estimators": 100,
        "max_depth": 20,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    
    def __init__(self, 
                 model_type: str = "logistic_regression",
                 tfidf_params: Optional[Dict] = None,
                 model_params: Optional[Dict] = None):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'logistic_regression' or 'random_forest'
            tfidf_params: Override TF-IDF vectorizer parameters
            model_params: Override model parameters
        """
        self.model_type = model_type
        
        # Merge custom params with defaults
        self.tfidf_params = {**self.DEFAULT_TFIDF_PARAMS, **(tfidf_params or {})}
        
        if model_type == "logistic_regression":
            default_params = self.DEFAULT_LR_PARAMS
        else:
            default_params = self.DEFAULT_RF_PARAMS
        self.model_params = {**default_params, **(model_params or {})}
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        
        # Separate models for category and priority
        self.category_model = self._create_model()
        self.priority_model = self._create_model()
        
        # Label encoders
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        
        # Training metrics
        self.metrics: Dict[str, Any] = {}
        self.is_fitted = False
    
    def _create_model(self):
        """Create the classification model based on model_type."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(**self.model_params)
        else:
            return RandomForestClassifier(**self.model_params)
    
    def fit(self, 
            X: pd.Series, 
            y_category: pd.Series,
            y_priority: pd.Series,
            test_size: float = 0.2,
            evaluate: bool = True) -> Dict[str, Any]:
        """
        Train the classifier on preprocessed text data.
        
        Args:
            X: Preprocessed text data
            y_category: Category labels
            y_priority: Priority labels
            test_size: Fraction of data for testing
            evaluate: Whether to evaluate and store metrics
            
        Returns:
            Dictionary with training results and metrics
        """
        # Encode labels
        y_cat_encoded = self.category_encoder.fit_transform(y_category)
        y_pri_encoded = self.priority_encoder.fit_transform(y_priority)
        
        # Split data
        X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = \
            train_test_split(
                X, y_cat_encoded, y_pri_encoded,
                test_size=test_size,
                random_state=42,
                stratify=y_cat_encoded
            )
        
        # Vectorize text
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train category model
        self.category_model.fit(X_train_tfidf, y_cat_train)
        
        # Train priority model
        self.priority_model.fit(X_train_tfidf, y_pri_train)
        
        self.is_fitted = True
        
        # Evaluate if requested
        if evaluate:
            self.metrics = self._evaluate(
                X_test_tfidf, 
                y_cat_test, y_pri_test,
                X_train_tfidf,
                y_cat_train, y_pri_train
            )
        
        return {
            "status": "success",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "categories": list(self.category_encoder.classes_),
            "priorities": list(self.priority_encoder.classes_),
            "metrics": self.metrics
        }
    
    def _evaluate(self, 
                  X_test, y_cat_test, y_pri_test,
                  X_train, y_cat_train, y_pri_train) -> Dict[str, Any]:
        """Evaluate model performance."""
        metrics = {}
        
        # Category predictions
        y_cat_pred = self.category_model.predict(X_test)
        y_cat_pred_train = self.category_model.predict(X_train)
        
        # Priority predictions
        y_pri_pred = self.priority_model.predict(X_test)
        y_pri_pred_train = self.priority_model.predict(X_train)
        
        # Category metrics
        metrics["category"] = {
            "train_accuracy": accuracy_score(y_cat_train, y_cat_pred_train),
            "test_accuracy": accuracy_score(y_cat_test, y_cat_pred),
            "precision": precision_score(y_cat_test, y_cat_pred, average="weighted"),
            "recall": recall_score(y_cat_test, y_cat_pred, average="weighted"),
            "f1_score": f1_score(y_cat_test, y_cat_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_cat_test, y_cat_pred).tolist(),
            "classification_report": classification_report(
                y_cat_test, y_cat_pred,
                target_names=self.category_encoder.classes_,
                output_dict=True
            )
        }
        
        # Priority metrics
        metrics["priority"] = {
            "train_accuracy": accuracy_score(y_pri_train, y_pri_pred_train),
            "test_accuracy": accuracy_score(y_pri_test, y_pri_pred),
            "precision": precision_score(y_pri_test, y_pri_pred, average="weighted"),
            "recall": recall_score(y_pri_test, y_pri_pred, average="weighted"),
            "f1_score": f1_score(y_pri_test, y_pri_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_pri_test, y_pri_pred).tolist(),
            "classification_report": classification_report(
                y_pri_test, y_pri_pred,
                target_names=self.priority_encoder.classes_,
                output_dict=True
            )
        }
        
        return metrics
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict category and priority for a single ticket.
        
        Args:
            text: Preprocessed ticket text
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Vectorize
        X = self.vectorizer.transform([text])
        
        # Get predictions
        cat_pred = self.category_model.predict(X)[0]
        pri_pred = self.priority_model.predict(X)[0]
        
        # Get probabilities
        cat_proba = self.category_model.predict_proba(X)[0]
        pri_proba = self.priority_model.predict_proba(X)[0]
        
        # Decode labels
        category = self.category_encoder.inverse_transform([cat_pred])[0]
        priority = self.priority_encoder.inverse_transform([pri_pred])[0]
        
        return {
            "category": category,
            "priority": priority,
            "confidence_category": float(np.max(cat_proba)),
            "confidence_priority": float(np.max(pri_proba)),
            "category_probabilities": {
                label: float(prob) 
                for label, prob in zip(self.category_encoder.classes_, cat_proba)
            },
            "priority_probabilities": {
                label: float(prob)
                for label, prob in zip(self.priority_encoder.classes_, pri_proba)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple tickets."""
        return [self.predict(text) for text in texts]
    
    def save(self, model_dir: str = "src/models/serialized") -> str:
        """
        Save model components to disk.
        
        Args:
            model_dir: Directory to save model files
            
        Returns:
            Path to saved model directory
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save all components
        joblib.dump(self.vectorizer, model_path / "vectorizer.joblib")
        joblib.dump(self.category_model, model_path / "category_model.joblib")
        joblib.dump(self.priority_model, model_path / "priority_model.joblib")
        joblib.dump(self.category_encoder, model_path / "category_encoder.joblib")
        joblib.dump(self.priority_encoder, model_path / "priority_encoder.joblib")
        joblib.dump(self.metrics, model_path / "metrics.joblib")
        joblib.dump({
            "model_type": self.model_type,
            "tfidf_params": self.tfidf_params,
            "model_params": self.model_params
        }, model_path / "config.joblib")
        
        print(f"✅ Model saved to {model_path}")
        return str(model_path)
    
    @classmethod
    def load(cls, model_dir: str = "src/models/serialized") -> "TicketClassifier":
        """
        Load model from disk.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            Loaded TicketClassifier instance
        """
        model_path = Path(model_dir)
        
        # Load config
        config = joblib.load(model_path / "config.joblib")
        
        # Create instance
        instance = cls(
            model_type=config["model_type"],
            tfidf_params=config["tfidf_params"],
            model_params=config["model_params"]
        )
        
        # Load components
        instance.vectorizer = joblib.load(model_path / "vectorizer.joblib")
        instance.category_model = joblib.load(model_path / "category_model.joblib")
        instance.priority_model = joblib.load(model_path / "priority_model.joblib")
        instance.category_encoder = joblib.load(model_path / "category_encoder.joblib")
        instance.priority_encoder = joblib.load(model_path / "priority_encoder.joblib")
        instance.metrics = joblib.load(model_path / "metrics.joblib")
        instance.is_fitted = True
        
        print(f"✅ Model loaded from {model_path}")
        return instance
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top important features for each category.
        
        Only available for Logistic Regression models.
        """
        if self.model_type != "logistic_regression":
            return {"error": "Feature importance only available for logistic regression"}
        
        feature_names = self.vectorizer.get_feature_names_out()
        importance = {}
        
        for idx, category in enumerate(self.category_encoder.classes_):
            coef = self.category_model.coef_[idx]
            top_indices = np.argsort(coef)[-top_n:][::-1]
            importance[category] = [
                (feature_names[i], float(coef[i])) 
                for i in top_indices
            ]
        
        return importance
    
    def print_metrics(self) -> None:
        """Print formatted evaluation metrics."""
        if not self.metrics:
            print("No metrics available. Train the model first.")
            return
        
        print("\n" + "=" * 60)
        print("📊 MODEL EVALUATION METRICS")
        print("=" * 60)
        
        for target in ["category", "priority"]:
            print(f"\n🎯 {target.upper()} CLASSIFICATION")
            print("-" * 40)
            m = self.metrics[target]
            print(f"  Train Accuracy: {m['train_accuracy']:.4f}")
            print(f"  Test Accuracy:  {m['test_accuracy']:.4f}")
            print(f"  Precision:      {m['precision']:.4f}")
            print(f"  Recall:         {m['recall']:.4f}")
            print(f"  F1 Score:       {m['f1_score']:.4f}")
        
        print("\n" + "=" * 60)


def main():
    """Train and evaluate the classifier."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.generator import TicketGenerator
    from src.data.preprocessing import TextPreprocessor
    
    print("🚀 Training Ticket Classifier\n")
    
    # Generate dataset
    print("📝 Generating dataset...")
    generator = TicketGenerator(seed=42)
    df = generator.generate_dataset(n_tickets=500, balanced=True)
    
    # Preprocess text
    print("🔧 Preprocessing text...")
    preprocessor = TextPreprocessor()
    df = preprocessor.combine_fields(df, ["subject", "description"], "text")
    df = preprocessor.preprocess_dataframe(df, "text", "processed_text")
    
    # Train classifier
    print("🤖 Training classifier...")
    classifier = TicketClassifier(model_type="logistic_regression")
    results = classifier.fit(
        X=df["processed_text"],
        y_category=df["category"],
        y_priority=df["priority"],
        test_size=0.2
    )
    
    # Print metrics
    classifier.print_metrics()
    
    # Save model
    print("\n💾 Saving model...")
    classifier.save()
    
    # Test prediction
    print("\n🔮 Test Prediction:")
    test_text = preprocessor.preprocess("I cannot login to my account and need urgent help")
    prediction = classifier.predict(test_text)
    print(f"  Text: 'I cannot login to my account and need urgent help'")
    print(f"  Category: {prediction['category']} ({prediction['confidence_category']:.2%})")
    print(f"  Priority: {prediction['priority']} ({prediction['confidence_priority']:.2%})")
    
    return classifier


if __name__ == "__main__":
    main()
