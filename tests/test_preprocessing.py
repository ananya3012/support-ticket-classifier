"""
Unit tests for text preprocessing module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True,
            min_token_length=2
        )
    
    def test_basic_preprocessing(self, preprocessor):
        """Test basic text preprocessing."""
        text = "I cannot login to my account!"
        result = preprocessor.preprocess(text)
        
        # Should be lowercase
        assert result == result.lower()
        # Should not contain punctuation
        assert "!" not in result
    
    def test_lowercase_conversion(self, preprocessor):
        """Test that text is converted to lowercase."""
        text = "URGENT: System DOWN!"
        result = preprocessor.preprocess(text)
        assert result == result.lower()
    
    def test_url_removal(self, preprocessor):
        """Test URL removal."""
        text = "Check https://example.com for details"
        result = preprocessor.preprocess(text)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_email_removal(self, preprocessor):
        """Test email address removal."""
        text = "Contact support@company.com for help"
        result = preprocessor.preprocess(text)
        assert "@" not in result
        assert "support@company.com" not in result
    
    def test_stopword_removal(self, preprocessor):
        """Test stopword removal."""
        text = "I want to get some help please"
        result = preprocessor.preprocess(text)
        
        # Common stopwords should be removed
        words = result.split()
        assert "i" not in words  # lowercase 'I' removed
        assert "to" not in words
        assert "some" not in words
    
    def test_keep_important_words(self, preprocessor):
        """Test that important words are kept."""
        text = "I cannot access my account"
        result = preprocessor.preprocess(text)
        
        # 'cannot' should be kept as it's semantically important
        assert "not" in result or "cannot" in result
    
    def test_empty_input(self, preprocessor):
        """Test handling of empty input."""
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess("   ") == ""
    
    def test_non_string_input(self, preprocessor):
        """Test handling of non-string input."""
        assert preprocessor.preprocess(None) == ""
        assert preprocessor.preprocess(123) == ""
    
    def test_batch_preprocessing(self, preprocessor):
        """Test batch preprocessing."""
        texts = [
            "First ticket about billing",
            "Second ticket about login",
            "Third ticket about bugs"
        ]
        results = preprocessor.preprocess_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
    
    def test_vocabulary_stats(self, preprocessor):
        """Test vocabulary statistics."""
        texts = [
            "billing issue payment",
            "billing refund request",
            "payment failed error"
        ]
        processed = preprocessor.preprocess_batch(texts)
        stats = preprocessor.get_vocabulary_stats(processed)
        
        assert "total_tokens" in stats
        assert "unique_tokens" in stats
        assert "most_common" in stats
        assert stats["total_tokens"] > 0


class TestPreprocessorConfiguration:
    """Test preprocessor configuration options."""
    
    def test_no_stopword_removal(self):
        """Test with stopword removal disabled."""
        preprocessor = TextPreprocessor(remove_stopwords=False)
        text = "I want to get help"
        result = preprocessor.preprocess(text)
        
        # More words should be kept
        assert len(result.split()) > 2
    
    def test_no_lemmatization(self):
        """Test with lemmatization disabled."""
        preprocessor = TextPreprocessor(lemmatize=False)
        text = "running quickly through issues"
        result = preprocessor.preprocess(text)
        
        # Words should not be lemmatized
        assert "running" in result or "quickly" in result
    
    def test_min_token_length(self):
        """Test minimum token length filter."""
        preprocessor = TextPreprocessor(min_token_length=4)
        text = "a b cd long word short"
        result = preprocessor.preprocess(text)
        
        # Short tokens should be filtered
        words = result.split()
        assert all(len(w) >= 4 for w in words if w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
