import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                           classification_report, precision_recall_curve)
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px

class ModelEvaluator:
    """Evaluate SDM model performance"""
    
    def __init__(self):
        self.metrics = {}
        self.results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive model metrics"""
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': tp / (tp + fn),  # True Positive Rate
            'specificity': tn / (tn + fp),  # True Negative Rate
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'tss': (tp / (tp + fn)) + (tn / (tn + fp)) - 1,  # True Skill Statistic
            'kappa': self._calculate_kappa(y_true, y_pred),
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        # ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        
        self.metrics = metrics
        return metrics
    
    def _calculate_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cohen's Kappa coefficient"""
        n = len(y_true)
        observed_agreement = np.sum(y_true == y_pred) / n
        
        # Expected agreement
        p_yes_true = np.sum(y_true == 1) / n
        p_yes_pred = np.sum(y_pred == 1) / n
        p_no_true = np.sum(y_true == 0) / n
        p_no_pred = np.sum(y_pred == 0) / n
        
        expected_agreement = (p_yes_true * p_yes_pred) + (p_no_true * p_no_pred)
        
        if expected_agreement == 1:
            return 1.0
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa
    
    def plot_roc_curve(self) -> go.Figure:
        """Plot interactive ROC curve"""
        if 'roc_curve' not in self.metrics:
            raise ValueError("Metrics not calculated yet")
        
        fpr = self.metrics['roc_curve']['fpr']
        tpr = self.metrics['roc_curve']['tpr']
        auc_score = self.metrics['auc']
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {auc_score:.3f})',
            line=dict(color='darkblue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            width=600,
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self) -> go.Figure:
        """Plot interactive confusion matrix"""
        if 'confusion_matrix' not in self.metrics:
            raise ValueError("Metrics not calculated yet")
        
        cm = self.metrics['confusion_matrix']
        cm_array = np.array([[cm['tn'], cm['fp']], 
                            [cm['fn'], cm['tp']]])
        
        fig = px.imshow(cm_array,
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Absence', 'Presence'],
                       y=['Absence', 'Presence'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        
        fig.update_layout(
            title='Confusion Matrix',
            width=500,
            height=400
        )
        
        return fig
    
    def plot_precision_recall_curve(self) -> go.Figure:
        """Plot interactive precision-recall curve"""
        if 'pr_curve' not in self.metrics:
            raise ValueError("Metrics not calculated yet")
        
        precision = self.metrics['pr_curve']['precision']
        recall = self.metrics['pr_curve']['recall']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='Precision-Recall curve',
            line=dict(color='darkgreen', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True,
            width=600,
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               top_n: int = 20) -> go.Figure:
        """Plot feature importance"""
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=max(400, top_n * 20),
            margin=dict(l=150)
        )
        
        return fig
    
    def plot_probability_distribution(self, y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray) -> go.Figure:
        """Plot probability distribution for presence/absence"""
        
        df = pd.DataFrame({
            'probability': y_pred_proba,
            'class': ['Absence' if y == 0 else 'Presence' for y in y_true]
        })
        
        fig = go.Figure()
        
        for class_name, color in [('Absence', 'blue'), ('Presence', 'red')]:
            class_data = df[df['class'] == class_name]['probability']
            fig.add_trace(go.Histogram(
                x=class_data,
                name=class_name,
                opacity=0.7,
                nbinsx=30,
                marker_color=color
            ))
        
        fig.update_layout(
            title='Probability Distribution by Class',
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            barmode='overlay',
            showlegend=True,
            width=700,
            height=500
        )
        
        return fig
    
    def create_metrics_summary(self) -> pd.DataFrame:
        """Create summary dataframe of all metrics"""
        if not self.metrics:
            raise ValueError("Metrics not calculated yet")
        
        summary_metrics = {k: v for k, v in self.metrics.items() 
                          if k not in ['confusion_matrix', 'roc_curve', 'pr_curve']}
        
        df = pd.DataFrame(list(summary_metrics.items()), 
                         columns=['Metric', 'Value'])
        df['Value'] = df['Value'].round(4)
        
        return df
    
    def generate_report(self, model_name: str = 'Random Forest SDM') -> Dict:
        """Generate comprehensive evaluation report"""
        if not self.metrics:
            raise ValueError("Metrics not calculated yet")
        
        report = {
            'model_name': model_name,
            'metrics_summary': self.create_metrics_summary(),
            'confusion_matrix': self.metrics['confusion_matrix'],
            'auc_score': self.metrics['auc'],
            'tss_score': self.metrics['tss'],
            'kappa_score': self.metrics['kappa'],
            'plots': {
                'roc_curve': self.plot_roc_curve(),
                'confusion_matrix': self.plot_confusion_matrix(),
                'pr_curve': self.plot_precision_recall_curve()
            }
        }
        
        return report

# Standalone functions for backward compatibility
def create_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> go.Figure:
    """Create ROC curve plot"""
    evaluator = ModelEvaluator()
    evaluator.calculate_metrics(y_true, np.round(y_pred_proba), y_pred_proba)
    return evaluator.plot_roc_curve()

def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Create confusion matrix plot"""
    evaluator = ModelEvaluator()
    evaluator.calculate_metrics(y_true, y_pred, np.ones_like(y_pred))
    return evaluator.plot_confusion_matrix()