"""
FastAPI Application for Ticket Classification

REST API endpoints for predicting ticket category and priority
with health checks and metrics reporting.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    TicketInput,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    BatchPredictionInput,
    BatchPredictionResponse
)
from src.data.preprocessing import TextPreprocessor
from src.models.classifier import TicketClassifier


# Global state
MODEL: Optional[TicketClassifier] = None
PREPROCESSOR: Optional[TextPreprocessor] = None
PREDICTION_COUNT: int = 0
START_TIME: datetime = datetime.now()


def load_model():
    """Load the trained model and preprocessor."""
    global MODEL, PREPROCESSOR
    
    model_path = PROJECT_ROOT / "src" / "models" / "serialized"
    
    if model_path.exists():
        try:
            MODEL = TicketClassifier.load(str(model_path))
            PREPROCESSOR = TextPreprocessor()
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return False
    else:
        print(f"⚠️ Model not found at {model_path}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Ticket Classification API...")
    load_model()
    yield
    # Shutdown
    print("👋 Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Ticket Classification API",
    description="""
    ## Intelligent Support Ticket Classification Platform
    
    This API provides machine learning-powered classification for customer support tickets,
    automatically predicting both **category** and **priority** levels.
    
    ### Features
    - 🎯 **Category Classification**: Billing, Technical, Account, Feature Request, General Inquiry
    - ⚡ **Priority Prediction**: Critical, High, Medium, Low
    - 📊 **Confidence Scores**: Probability distribution for all classes
    - 🔄 **Batch Processing**: Classify multiple tickets in one request
    
    ### Model Details
    - Algorithm: TF-IDF + Logistic Regression
    - Training Data: 500+ labeled support tickets
    - Accuracy: ~85%+ for category, ~70%+ for priority
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for frontend access


frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5500"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],  # Configure appropriately for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "static"),
    name="static"
)


@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive frontend."""
    static_path = PROJECT_ROOT / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "Welcome to Ticket Classification API", "docs": "/docs"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check API health status and model availability"
)
async def health_check():
    """
    Health check endpoint for monitoring and container orchestration.
    
    Returns:
        Health status, model loaded state, and API version
    """
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None and MODEL.is_fitted,
        version="1.0.0"
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="Model Metrics",
    description="Get model performance metrics and usage statistics"
)
async def get_metrics():
    """
    Get model evaluation metrics and prediction statistics.
    
    Returns:
        Model accuracy, F1 scores, and prediction count
    """
    if MODEL is None or not MODEL.metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded or metrics not available"
        )
    
    return MetricsResponse(
        category_accuracy=MODEL.metrics["category"]["test_accuracy"],
        priority_accuracy=MODEL.metrics["priority"]["test_accuracy"],
        category_f1=MODEL.metrics["category"]["f1_score"],
        priority_f1=MODEL.metrics["priority"]["f1_score"],
        total_predictions=PREDICTION_COUNT
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    tags=["Prediction"],
    summary="Classify Single Ticket",
    description="Predict category and priority for a single support ticket"
)
async def predict_ticket(ticket: TicketInput):
    """
    Predict category and priority for a single ticket.
    
    Args:
        ticket: TicketInput with subject and description
        
    Returns:
        PredictionResponse with category, priority, and confidence scores
    """
    global PREDICTION_COUNT
    
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Combine and preprocess text
        combined_text = f"{ticket.subject} {ticket.description}"
        processed_text = PREPROCESSOR.preprocess(combined_text)
        
        # Get prediction
        prediction = MODEL.predict(processed_text)
        
        PREDICTION_COUNT += 1
        
        return PredictionResponse(
            ticket_text=combined_text,
            category=prediction["category"],
            priority=prediction["priority"],
            confidence_category=prediction["confidence_category"],
            confidence_priority=prediction["confidence_priority"],
            category_probabilities=prediction["category_probabilities"],
            priority_probabilities=prediction["priority_probabilities"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    tags=["Prediction"],
    summary="Classify Multiple Tickets",
    description="Predict category and priority for multiple tickets in batch"
)
async def predict_batch(batch: BatchPredictionInput):
    """
    Predict category and priority for multiple tickets.
    
    Args:
        batch: BatchPredictionInput with list of tickets
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    global PREDICTION_COUNT
    
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        predictions = []
        
        for ticket in batch.tickets:
            combined_text = f"{ticket.subject} {ticket.description}"
            processed_text = PREPROCESSOR.preprocess(combined_text)
            prediction = MODEL.predict(processed_text)
            
            predictions.append(PredictionResponse(
                ticket_text=combined_text,
                category=prediction["category"],
                priority=prediction["priority"],
                confidence_category=prediction["confidence_category"],
                confidence_priority=prediction["confidence_priority"],
                category_probabilities=prediction["category_probabilities"],
                priority_probabilities=prediction["priority_probabilities"]
            ))
            
            PREDICTION_COUNT += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/categories",
    tags=["Reference"],
    summary="List Categories",
    description="Get list of all supported ticket categories"
)
async def list_categories():
    """Get all supported ticket categories."""
    return {
        "categories": [
            {"name": "Billing", "description": "Payment issues, invoices, refunds"},
            {"name": "Technical", "description": "Product bugs, errors, technical problems"},
            {"name": "Account", "description": "Login issues, profile, settings"},
            {"name": "Feature Request", "description": "New feature suggestions"},
            {"name": "General Inquiry", "description": "General questions and information"}
        ]
    }


@app.get(
    "/priorities",
    tags=["Reference"],
    summary="List Priorities",
    description="Get list of all priority levels with descriptions"
)
async def list_priorities():
    """Get all priority levels and their descriptions."""
    return {
        "priorities": [
            {"name": "Critical", "description": "System down, security issues", "sla_hours": 1},
            {"name": "High", "description": "Major functionality broken", "sla_hours": 4},
            {"name": "Medium", "description": "Inconvenience but workaround exists", "sla_hours": 24},
            {"name": "Low", "description": "Minor issues, questions", "sla_hours": 72}
        ]
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
