"""
Pydantic Schemas for API Request/Response Validation

Defines data models for ticket prediction requests and responses.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """
    Input schema for ticket prediction.
    
    Attributes:
        subject: Ticket subject line
        description: Detailed ticket description
    """
    subject: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Ticket subject line",
        examples=["Cannot login to my account"]
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Detailed ticket description",
        examples=["I've been trying to login for the past hour but keep getting an error message."]
    )


class PredictionResponse(BaseModel):
    """
    Response schema for ticket prediction.
    
    Attributes:
        ticket_text: Combined input text
        category: Predicted category
        priority: Predicted priority
        confidence_category: Confidence score for category prediction (0-1)
        confidence_priority: Confidence score for priority prediction (0-1)
        category_probabilities: Probability distribution over all categories
        priority_probabilities: Probability distribution over all priorities
    """
    ticket_text: str = Field(
        ...,
        description="Combined subject and description text"
    )
    category: str = Field(
        ...,
        description="Predicted ticket category",
        examples=["Account", "Billing", "Technical", "Feature Request", "General Inquiry"]
    )
    priority: str = Field(
        ...,
        description="Predicted ticket priority",
        examples=["Critical", "High", "Medium", "Low"]
    )
    confidence_category: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for category prediction"
    )
    confidence_priority: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for priority prediction"
    )
    category_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution over all categories"
    )
    priority_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution over all priorities"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(
        default="healthy",
        description="API health status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready"
    )
    version: str = Field(
        ...,
        description="API version"
    )


class MetricsResponse(BaseModel):
    """Model metrics response schema."""
    category_accuracy: float = Field(
        ...,
        description="Category classification accuracy"
    )
    priority_accuracy: float = Field(
        ...,
        description="Priority classification accuracy"
    )
    category_f1: float = Field(
        ...,
        description="Category F1 score (weighted)"
    )
    priority_f1: float = Field(
        ...,
        description="Priority F1 score (weighted)"
    )
    total_predictions: int = Field(
        default=0,
        description="Total number of predictions made"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    tickets: list[TicketInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of tickets to predict"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    count: int = Field(
        ...,
        description="Number of predictions returned"
    )
