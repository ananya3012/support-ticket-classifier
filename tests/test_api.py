"""
Integration tests for FastAPI endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app, MODEL, PREPROCESSOR


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_categories_endpoint(self, client):
        """Test categories listing endpoint."""
        response = client.get("/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) == 5
        
        # Check category structure
        for category in data["categories"]:
            assert "name" in category
            assert "description" in category
    
    def test_priorities_endpoint(self, client):
        """Test priorities listing endpoint."""
        response = client.get("/priorities")
        assert response.status_code == 200
        
        data = response.json()
        assert "priorities" in data
        assert len(data["priorities"]) == 4
        
        # Check priority structure
        for priority in data["priorities"]:
            assert "name" in priority
            assert "description" in priority
            assert "sla_hours" in priority
    
    def test_prediction_endpoint_validation(self, client):
        """Test prediction endpoint input validation."""
        # Missing fields
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Subject too short
        response = client.post("/predict", json={
            "subject": "ab",
            "description": "This is a valid description for testing"
        })
        assert response.status_code == 422
        
        # Description too short
        response = client.post("/predict", json={
            "subject": "Valid subject line",
            "description": "Short"
        })
        assert response.status_code == 422
    
    def test_prediction_endpoint_success(self, client):
        """Test successful prediction (requires trained model)."""
        response = client.post("/predict", json={
            "subject": "Cannot login to my account",
            "description": "I've been trying to login for an hour but keep getting an error message about invalid credentials"
        })
        
        # If model is loaded, should return 200
        # If model not loaded, should return 503
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "category" in data
            assert "priority" in data
            assert "confidence_category" in data
            assert "confidence_priority" in data
    
    def test_batch_prediction_endpoint(self, client):
        """Test batch prediction endpoint."""
        response = client.post("/predict/batch", json={
            "tickets": [
                {
                    "subject": "Billing question",
                    "description": "I was charged twice for my subscription this month"
                },
                {
                    "subject": "Feature request",
                    "description": "Would love to see dark mode in the next update"
                }
            ]
        })
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "count" in data
            assert data["count"] == 2
    
    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestAPIEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_long_text_input(self, client):
        """Test handling of very long text input."""
        long_description = "This is a test. " * 100
        
        response = client.post("/predict", json={
            "subject": "Test ticket",
            "description": long_description
        })
        
        # Should not crash, either success or model not loaded
        assert response.status_code in [200, 503]
    
    def test_special_characters(self, client):
        """Test handling of special characters."""
        response = client.post("/predict", json={
            "subject": "Test with émojis 🎉 and spëcial chars",
            "description": "Description with <html> tags & symbols @#$%"
        })
        
        assert response.status_code in [200, 503]
    
    def test_unicode_input(self, client):
        """Test handling of Unicode input."""
        response = client.post("/predict", json={
            "subject": "日本語テスト",
            "description": "This ticket contains Unicode: 中文, العربية, עברית"
        })
        
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
