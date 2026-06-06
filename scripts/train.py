#!/usr/bin/env python3
"""
Training Script

Generates dataset, trains the classifier, creates visualizations,
and saves the model for API deployment.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator import TicketGenerator
from src.data.database import TicketDatabase
from src.data.preprocessing import TextPreprocessor
from src.models.classifier import TicketClassifier
from src.visualizations.plots import VisualizationManager


def main():
    """Complete training pipeline."""
    print("\n" + "=" * 60)
    print("🚀 TICKET CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # ===== Step 1: Generate Dataset =====
    print("\n📝 Step 1: Generating Dataset...")
    generator = TicketGenerator(seed=42)
    df = generator.generate_dataset(n_tickets=500, balanced=True)
    
    # Save to CSV
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "tickets.csv"
    generator.save_to_csv(df, str(csv_path))
    
    print(f"   Generated {len(df)} tickets")
    print(f"   Categories: {df['category'].nunique()}")
    print(f"   Priorities: {df['priority'].nunique()}")
    
    # ===== Step 2: Store in Database =====
    print("\n💾 Step 2: Storing in Database...")
    db_path = data_dir / "tickets.db"
    with TicketDatabase(str(db_path)) as db:
        db.create_schema()
        count = db.load_from_csv(str(csv_path))
        print(f"   Stored {count} tickets in SQLite")
    
    # ===== Step 3: Preprocess Text =====
    print("\n🔧 Step 3: Preprocessing Text...")
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        min_token_length=2
    )
    
    # Combine subject and description
    df = preprocessor.combine_fields(df, ["subject", "description"], "text")
    
    # Preprocess text
    df = preprocessor.preprocess_dataframe(df, "text", "processed_text")
    
    # Get vocabulary stats
    stats = preprocessor.get_vocabulary_stats(df["processed_text"].tolist())
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Unique tokens: {stats['unique_tokens']}")
    print(f"   Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    
    # ===== Step 4: Train Classifier =====
    print("\n🤖 Step 4: Training Classifier...")
    classifier = TicketClassifier(model_type="logistic_regression")
    
    results = classifier.fit(
        X=df["processed_text"],
        y_category=df["category"],
        y_priority=df["priority"],
        test_size=0.2,
        evaluate=True
    )
    
    print(f"   Train size: {results['train_size']}")
    print(f"   Test size: {results['test_size']}")
    print(f"   Categories: {results['categories']}")
    print(f"   Priorities: {results['priorities']}")
    
    # Print detailed metrics
    classifier.print_metrics()
    
    # ===== Step 5: Generate Visualizations =====
    print("\n📊 Step 5: Generating Visualizations...")
    viz_dir = PROJECT_ROOT / "visualizations"
    viz = VisualizationManager(output_dir=str(viz_dir))
    
    viz.create_all_visualizations(
        df=df,
        metrics=classifier.metrics,
        show=False
    )
    
    # ===== Step 6: Save Model =====
    print("\n💾 Step 6: Saving Model...")
    model_dir = PROJECT_ROOT / "src" / "models" / "serialized"
    classifier.save(str(model_dir))
    
    # ===== Step 7: Test Prediction =====
    print("\n🔮 Step 7: Testing Prediction...")
    test_cases = [
        ("Cannot login to account", "I forgot my password and reset email not working"),
        ("Charged twice", "I was charged $99 twice on my credit card for one subscription"),
        ("Feature request", "Would love to see dark mode added to the app"),
        ("App crashes", "The application crashes every time I click export button"),
        ("Pricing question", "What are the differences between Pro and Enterprise plans?")
    ]
    
    for subject, description in test_cases:
        combined = f"{subject} {description}"
        processed = preprocessor.preprocess(combined)
        prediction = classifier.predict(processed)
        
        print(f"\n   📩 \"{subject[:30]}...\"")
        print(f"      Category: {prediction['category']} ({prediction['confidence_category']:.1%})")
        print(f"      Priority: {prediction['priority']} ({prediction['confidence_priority']:.1%})")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📁 Outputs:")
    print(f"   Dataset:        {csv_path}")
    print(f"   Database:       {db_path}")
    print(f"   Model:          {model_dir}")
    print(f"   Visualizations: {viz_dir}")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Start API:   uvicorn src.api.main:app --reload")
    print(f"   2. Open docs:   http://localhost:8000/docs")
    print(f"   3. Run tests:   pytest tests/ -v")
    print()
    
    return classifier


if __name__ == "__main__":
    main()
