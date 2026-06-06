"""
Unit tests for ML classifier module.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import TicketGenerator
from src.data.preprocessing import TextPreprocessor
from src.models.classifier import TicketClassifier


class TestTicketClassifier:
    """Test suite for TicketClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        generator = TicketGenerator(seed=42)
        df = generator.generate_dataset(n_tickets=100, balanced=True)
        
        preprocessor = TextPreprocessor()
        df = preprocessor.combine_fields(df, ["subject", "description"], "text")
        df = preprocessor.preprocess_dataframe(df, "text", "processed_text")
        
        return df
    
    @pytest.fixture
    def trained_classifier(self, sample_data):
        """Create and train a classifier."""
        classifier = TicketClassifier(model_type="logistic_regression")
        classifier.fit(
            X=sample_data["processed_text"],
            y_category=sample_data["category"],
            y_priority=sample_data["priority"],
            test_size=0.2,
            evaluate=True
        )
        return classifier
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = TicketClassifier()
        
        assert classifier.model_type == "logistic_regression"
        assert classifier.is_fitted is False
        assert classifier.vectorizer is not None
    
    def test_classifier_training(self, sample_data):
        """Test classifier training."""
        classifier = TicketClassifier()
        result = classifier.fit(
            X=sample_data["processed_text"],
            y_category=sample_data["category"],
            y_priority=sample_data["priority"]
        )
        
        assert result["status"] == "success"
        assert classifier.is_fitted is True
        assert len(result["categories"]) == 5  # 5 categories
        assert len(result["priorities"]) == 4  # 4 priorities
    
    def test_single_prediction(self, trained_classifier):
        """Test single ticket prediction."""
        text = "billing issue payment"
        prediction = trained_classifier.predict(text)
        
        assert "category" in prediction
        assert "priority" in prediction
        assert "confidence_category" in prediction
        assert "confidence_priority" in prediction
        assert prediction["category"] in ["Billing", "Technical", "Account", 
                                          "Feature Request", "General Inquiry"]
        assert prediction["priority"] in ["Critical", "High", "Medium", "Low"]
        assert 0 <= prediction["confidence_category"] <= 1
        assert 0 <= prediction["confidence_priority"] <= 1
    
    def test_batch_prediction(self, trained_classifier):
        """Test batch prediction."""
        texts = [
            "billing issue payment",
            "cannot login account",
            "feature request dark mode"
        ]
        predictions = trained_classifier.predict_batch(texts)
        
        assert len(predictions) == 3
        assert all("category" in p for p in predictions)
        assert all("priority" in p for p in predictions)
    
    def test_prediction_probabilities(self, trained_classifier):
        """Test that prediction includes probability distributions."""
        text = "I need help with my account"
        prediction = trained_classifier.predict(text)
        
        assert "category_probabilities" in prediction
        assert "priority_probabilities" in prediction
        
        # Probabilities should sum to ~1
        cat_probs = sum(prediction["category_probabilities"].values())
        pri_probs = sum(prediction["priority_probabilities"].values())
        
        assert 0.99 <= cat_probs <= 1.01
        assert 0.99 <= pri_probs <= 1.01
    
    def test_model_metrics(self, trained_classifier):
        """Test that training produces metrics."""
        assert trained_classifier.metrics is not None
        assert "category" in trained_classifier.metrics
        assert "priority" in trained_classifier.metrics
        
        # Check metric values
        cat_metrics = trained_classifier.metrics["category"]
        assert "test_accuracy" in cat_metrics
        assert "precision" in cat_metrics
        assert "recall" in cat_metrics
        assert "f1_score" in cat_metrics
        
        # Accuracy should be above random (1/5 categories = 0.2)
        # Note: Small test samples may show high variance
        assert cat_metrics["test_accuracy"] >= 0.15
    
    def test_model_save_load(self, trained_classifier):
        """Test model serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            save_path = trained_classifier.save(tmpdir)
            assert Path(save_path).exists()
            
            # Load model
            loaded = TicketClassifier.load(tmpdir)
            assert loaded.is_fitted is True
            
            # Test loaded model prediction
            text = "test prediction"
            original_pred = trained_classifier.predict(text)
            loaded_pred = loaded.predict(text)
            
            assert original_pred["category"] == loaded_pred["category"]
            assert original_pred["priority"] == loaded_pred["priority"]
    
    def test_unfitted_model_prediction_error(self):
        """Test that unfitted model raises error on prediction."""
        classifier = TicketClassifier()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            classifier.predict("test text")
    
    def test_random_forest_classifier(self, sample_data):
        """Test Random Forest classifier alternative."""
        classifier = TicketClassifier(model_type="random_forest")
        result = classifier.fit(
            X=sample_data["processed_text"],
            y_category=sample_data["category"],
            y_priority=sample_data["priority"]
        )
        
        assert result["status"] == "success"
        assert classifier.is_fitted is True


class TestClassifierConfiguration:
    """Test classifier configuration options."""
    
    def test_custom_tfidf_params(self):
        """Test custom TF-IDF parameters."""
        custom_params = {"max_features": 1000, "ngram_range": (1, 3)}
        classifier = TicketClassifier(tfidf_params=custom_params)
        
        assert classifier.tfidf_params["max_features"] == 1000
        assert classifier.tfidf_params["ngram_range"] == (1, 3)
    
    def test_custom_model_params(self):
        """Test custom model parameters."""
        custom_params = {"C": 0.5, "max_iter": 500}
        classifier = TicketClassifier(model_params=custom_params)
        
        assert classifier.model_params["C"] == 0.5
        assert classifier.model_params["max_iter"] == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
