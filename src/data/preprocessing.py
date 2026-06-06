"""
Text Preprocessing Pipeline

Provides text cleaning and NLP preprocessing for ticket classification,
including lowercasing, stopword removal, tokenization, and lemmatization.
"""

import re
import string
from typing import List, Optional, Union
import numpy as np
import pandas as pd

# NLTK imports with download handling
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    # Ensure required NLTK data is available
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Text preprocessing pipeline for support ticket classification.
    
    Processing steps:
        1. Lowercase text
        2. Remove special characters and numbers
        3. Remove punctuation
        4. Tokenize text
        5. Remove stopwords
        6. Lemmatize tokens
        7. Join tokens back to text
    
    This preprocessing is optimized for support ticket text which often contains:
        - Technical jargon
        - Error codes
        - Email addresses
        - URLs
    """
    
    # Domain-specific stopwords to remove
    DOMAIN_STOPWORDS = {
        "please", "help", "need", "want", "would", "like", "thanks",
        "thank", "hi", "hello", "hey", "regards", "best", "sincerely",
        "dear", "sir", "madam", "team", "support"
    }
    
    # Words to keep even if they're usually stopwords (important for tickets)
    KEEP_WORDS = {
        "not", "no", "cannot", "can't", "won't", "don't", "doesn't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
        "error", "issue", "problem", "bug", "broken", "failed", "failing"
    }
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 min_token_length: int = 2,
                 remove_numbers: bool = False,
                 custom_stopwords: Optional[set] = None):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
            min_token_length: Minimum token length to keep
            remove_numbers: Whether to remove numeric tokens
            custom_stopwords: Additional stopwords to remove
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self.remove_numbers = remove_numbers
        
        # Build stopword set
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(self.DOMAIN_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        # Remove words we want to keep
        self.stopwords -= self.KEEP_WORDS
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Steps:
            - Convert to lowercase
            - Remove URLs
            - Remove email addresses
            - Remove special characters (keep spaces)
            - Remove extra whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove order/invoice IDs (common in ticket text)
        text = re.sub(r'#?\d{5,}', '', text)
        text = re.sub(r'tkt-\w+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'inv-\w+', '', text, flags=re.IGNORECASE)
        
        # Remove punctuation (keep apostrophes for contractions)
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Remove numbers if configured
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split if NLTK tokenizer fails
            tokens = text.split()
        return tokens
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [t for t in tokens if t not in self.stopwords]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base form."""
        if not self.lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens by length and content."""
        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            # Skip tokens that are only punctuation or numbers
            if token.isdigit() and self.remove_numbers:
                continue
            # Skip tokens with only non-alphanumeric chars
            if not any(c.isalnum() for c in token):
                continue
            filtered.append(token)
        return filtered
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw text input
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self._clean_text(text)
        
        if not text:
            return ""
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self._lemmatize_tokens(tokens)
        
        # Filter tokens
        tokens = self._filter_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            List of preprocessed strings
        """
        return [self.preprocess(text) for text in texts]
    
    def combine_fields(self, 
                       df: pd.DataFrame,
                       fields: List[str] = ["subject", "description"],
                       output_field: str = "text") -> pd.DataFrame:
        """
        Combine multiple text fields into one.
        
        Args:
            df: DataFrame with text columns
            fields: List of column names to combine
            output_field: Name of output column
            
        Returns:
            DataFrame with new combined column
        """
        df = df.copy()
        df[output_field] = df[fields].fillna('').agg(' '.join, axis=1)
        return df
    
    def preprocess_dataframe(self,
                             df: pd.DataFrame,
                             text_column: str = "text",
                             output_column: str = "processed_text") -> pd.DataFrame:
        """
        Preprocess text column in DataFrame.
        
        Args:
            df: DataFrame with text column
            text_column: Name of input text column
            output_column: Name of output column
            
        Returns:
            DataFrame with new preprocessed column
        """
        df = df.copy()
        df[output_column] = df[text_column].apply(self.preprocess)
        return df
    
    def get_vocabulary_stats(self, texts: List[str]) -> dict:
        """
        Get vocabulary statistics from preprocessed texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Dictionary with vocabulary statistics
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
        
        unique_tokens = set(all_tokens)
        
        # Count token frequencies
        from collections import Counter
        token_counts = Counter(all_tokens)
        
        return {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(unique_tokens),
            "avg_tokens_per_doc": np.mean([len(t.split()) for t in texts]),
            "most_common": token_counts.most_common(20)
        }


def main():
    """Demo preprocessing pipeline."""
    # Sample ticket texts
    sample_texts = [
        "I cannot login to my account! Please help urgently.",
        "Hi team, I was charged $99.99 twice for order #123456. Please refund.",
        "The API endpoint /api/users returns 500 error. Check server logs.",
        "Would love to see dark mode in the next update. Thanks!",
        "How do I export my data to CSV format? Need this for reporting."
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        min_token_length=2
    )
    
    print("📝 Text Preprocessing Demo\n")
    print("-" * 60)
    
    for text in sample_texts:
        processed = preprocessor.preprocess(text)
        print(f"Original:   {text}")
        print(f"Processed:  {processed}")
        print("-" * 60)
    
    # Get vocabulary stats
    processed_texts = preprocessor.preprocess_batch(sample_texts)
    stats = preprocessor.get_vocabulary_stats(processed_texts)
    
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Unique tokens: {stats['unique_tokens']}")
    print(f"   Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    print(f"   Top 10 tokens: {stats['most_common'][:10]}")


if __name__ == "__main__":
    main()
