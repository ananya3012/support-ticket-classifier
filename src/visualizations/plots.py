"""
Visualization Module

Creates data distribution and model evaluation visualizations
using Matplotlib for analysis and reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


class VisualizationManager:
    """
    Creates publication-quality visualizations for ticket analysis.
    
    Visualization types:
        - Category distribution (bar chart)
        - Priority distribution (pie chart)
        - Confusion matrix (heatmap)
        - Feature importance (horizontal bar)
        - Model comparison (grouped bar)
    """
    
    # Color palettes
    CATEGORY_COLORS = {
        "Billing": "#3498db",
        "Technical": "#e74c3c",
        "Account": "#9b59b6",
        "Feature Request": "#2ecc71",
        "General Inquiry": "#f39c12"
    }
    
    PRIORITY_COLORS = {
        "Critical": "#c0392b",
        "High": "#e74c3c",
        "Medium": "#f39c12",
        "Low": "#27ae60"
    }
    
    # Style settings
    STYLE_CONFIG = {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100
    }
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualization manager.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply style
        plt.rcParams.update(self.STYLE_CONFIG)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_category_distribution(self, 
                                   df: pd.DataFrame,
                                   column: str = "category",
                                   title: str = "Ticket Category Distribution",
                                   save: bool = True) -> plt.Figure:
        """
        Create bar chart showing category distribution.
        
        Args:
            df: DataFrame with category column
            column: Name of category column
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Count categories
        counts = df[column].value_counts()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars with custom colors
        colors = [self.CATEGORY_COLORS.get(cat, "#7f8c8d") for cat in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        # Customize
        ax.set_xlabel("Category", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Tickets", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Rotate x labels if needed
        plt.xticks(rotation=15, ha='right')
        
        # Add percentage annotation
        total = counts.sum()
        for bar, count in zip(bars, counts.values):
            pct = count / total * 100
            ax.annotate(f'({pct:.1f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                       ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "category_distribution.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")
        
        return fig
    
    def plot_priority_distribution(self,
                                   df: pd.DataFrame,
                                   column: str = "priority",
                                   title: str = "Ticket Priority Distribution",
                                   save: bool = True) -> plt.Figure:
        """
        Create pie chart showing priority distribution.
        
        Args:
            df: DataFrame with priority column
            column: Name of priority column
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Count priorities - reorder to Critical > High > Medium > Low
        priority_order = ["Critical", "High", "Medium", "Low"]
        counts = df[column].value_counts()
        counts = counts.reindex([p for p in priority_order if p in counts.index])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Colors
        colors = [self.PRIORITY_COLORS.get(p, "#95a5a6") for p in counts.index]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(counts),
            shadow=True,
            textprops={'fontsize': 11}
        )
        
        # Style autotexts
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with counts
        legend_labels = [f"{p}: {c}" for p, c in zip(counts.index, counts.values)]
        ax.legend(wedges, legend_labels, title="Priority (Count)",
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "priority_distribution.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")
        
        return fig
    
    def plot_confusion_matrix(self,
                              confusion_matrix: np.ndarray,
                              labels: List[str],
                              title: str = "Confusion Matrix",
                              target: str = "category",
                              save: bool = True) -> plt.Figure:
        """
        Create heatmap visualization of confusion matrix.
        
        Args:
            confusion_matrix: Numpy array of confusion matrix
            labels: Class labels
            title: Plot title
            target: Target name for filename ('category' or 'priority')
            save: Whether to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, label='Proportion')
        
        # Labels
        ax.set(xticks=np.arange(len(labels)),
               yticks=np.arange(len(labels)),
               xticklabels=labels,
               yticklabels=labels,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                # Show both count and percentage
                text = f"{confusion_matrix[i, j]}\n({cm_normalized[i, j]:.1%})"
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"confusion_matrix_{target}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")
        
        return fig
    
    def plot_metrics_comparison(self,
                                metrics: Dict[str, Dict],
                                title: str = "Model Performance Metrics",
                                save: bool = True) -> plt.Figure:
        """
        Create grouped bar chart comparing metrics for category and priority.
        
        Args:
            metrics: Dictionary with 'category' and 'priority' metrics
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        x = np.arange(len(metric_names))
        width = 0.35
        
        category_vals = [
            metrics['category']['test_accuracy'],
            metrics['category']['precision'],
            metrics['category']['recall'],
            metrics['category']['f1_score']
        ]
        
        priority_vals = [
            metrics['priority']['test_accuracy'],
            metrics['priority']['precision'],
            metrics['priority']['recall'],
            metrics['priority']['f1_score']
        ]
        
        bars1 = ax.bar(x - width/2, category_vals, width, label='Category', color='#3498db')
        bars2 = ax.bar(x + width/2, priority_vals, width, label='Priority', color='#e74c3c')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "metrics_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")
        
        return fig
    
    def plot_feature_importance(self,
                                importance: Dict[str, List[Tuple[str, float]]],
                                top_n: int = 10,
                                title: str = "Top Features by Category",
                                save: bool = True) -> plt.Figure:
        """
        Create horizontal bar chart showing top features per category.
        
        Args:
            importance: Dictionary mapping categories to (feature, importance) tuples
            top_n: Number of top features to show
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        n_categories = len(importance)
        fig, axes = plt.subplots(1, n_categories, figsize=(4 * n_categories, 6))
        
        if n_categories == 1:
            axes = [axes]
        
        for ax, (category, features) in zip(axes, importance.items()):
            # Get top N features
            top_features = features[:top_n]
            labels = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            
            # Create horizontal bar chart
            color = self.CATEGORY_COLORS.get(category, "#7f8c8d")
            y_pos = np.arange(len(labels))
            
            ax.barh(y_pos, values, color=color, edgecolor='white')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(category, fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "feature_importance.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")
        
        return fig
    
    def create_all_visualizations(self,
                                  df: pd.DataFrame,
                                  metrics: Optional[Dict] = None,
                                  show: bool = False) -> None:
        """
        Generate all standard visualizations.
        
        Args:
            df: DataFrame with ticket data
            metrics: Model metrics dictionary (optional)
            show: Whether to display plots
        """
        print("\n📊 Generating Visualizations...\n")
        
        # Distribution plots
        self.plot_category_distribution(df)
        self.plot_priority_distribution(df)
        
        # Model metrics if available
        if metrics:
            # Confusion matrices
            if 'category' in metrics:
                cm_cat = np.array(metrics['category']['confusion_matrix'])
                labels_cat = list(metrics['category']['classification_report'].keys())
                labels_cat = [l for l in labels_cat if l not in ['accuracy', 'macro avg', 'weighted avg']]
                self.plot_confusion_matrix(cm_cat, labels_cat, 
                                          "Category Classification - Confusion Matrix",
                                          "category")
            
            if 'priority' in metrics:
                cm_pri = np.array(metrics['priority']['confusion_matrix'])
                labels_pri = list(metrics['priority']['classification_report'].keys())
                labels_pri = [l for l in labels_pri if l not in ['accuracy', 'macro avg', 'weighted avg']]
                self.plot_confusion_matrix(cm_pri, labels_pri,
                                          "Priority Classification - Confusion Matrix",
                                          "priority")
            
            # Metrics comparison
            self.plot_metrics_comparison(metrics)
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
        
        if show:
            plt.show()


def main():
    """Demo visualization generation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.generator import TicketGenerator
    
    # Generate sample data
    generator = TicketGenerator(seed=42)
    df = generator.generate_dataset(n_tickets=500, balanced=True)
    
    # Create visualizations
    viz = VisualizationManager()
    viz.plot_category_distribution(df)
    viz.plot_priority_distribution(df)
    
    print("\n✅ Visualizations generated successfully!")


if __name__ == "__main__":
    main()
