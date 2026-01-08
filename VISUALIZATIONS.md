# Model Visualization Guide

## Overview

After model training, the pipeline automatically generates comprehensive visualizations showing:
- Model comparison across all trained models
- Best model performance metrics
- Learning curves
- Confusion matrix
- Performance metrics comparison

## Generated Visualizations

All visualizations are saved in `models/visualizations/` directory:

### 1. **model_comparison.png**
- Compares all trained models
- Shows Test Accuracy vs CV Mean Accuracy
- Bar chart with sorted models by performance
- Helps identify best performing models

### 2. **best_model_performance.png**
- Two-panel visualization:
  - **Left**: Test Accuracy vs CV Mean Accuracy with error bars
  - **Right**: Cross-validation scores distribution (box plot)
- Shows consistency of model performance

### 3. **learning_curve.png**
- Shows how model performance changes with training set size
- Training score vs Validation score
- Helps identify overfitting/underfitting
- Shows model's learning capacity

### 4. **confusion_matrix.png**
- Heatmap of confusion matrix
- Shows True Positives, True Negatives, False Positives, False Negatives
- Color-coded for easy interpretation

### 5. **metrics_comparison.png**
- Horizontal bar chart comparing all metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- Color-coded for visual appeal

## File Locations

```
models/
├── best_model.joblib          # Best performing model (with metadata)
├── dimension_reducer.joblib   # Feature reducer
├── evaluation_report.json     # Detailed evaluation metrics
├── model_comparison.json      # Comparison of all models
└── visualizations/
    ├── model_comparison.png
    ├── best_model_performance.png
    ├── learning_curve.png
    ├── confusion_matrix.png
    └── metrics_comparison.png
```

## Model Artifact Structure

The `best_model.joblib` file contains:

```python
{
    'model': <trained_model_object>,
    'model_name': 'gradient_boosting',  # Name of best model
    'scaler': <StandardScaler_object>,  # Feature scaler
    'feature_names': [...],              # List of feature names
    'test_accuracy': 0.8523,            # Test set accuracy
    'cv_mean_accuracy': 0.8489,          # CV mean accuracy
    'cv_std_accuracy': 0.0123,          # CV std accuracy
    'best_parameters': {...},            # Best hyperparameters
    'training_info': {...}               # Training metadata
}
```

## Viewing Visualizations

### Option 1: Direct File Access
```bash
# Navigate to visualizations directory
cd models/visualizations/

# Open images (platform-specific)
# Windows:
start model_comparison.png
# Linux:
xdg-open model_comparison.png
# Mac:
open model_comparison.png
```

### Option 2: Python Script
```python
from PIL import Image
import matplotlib.pyplot as plt

# Load and display
img = Image.open('models/visualizations/best_model_performance.png')
plt.figure(figsize=(16, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## Interpreting Visualizations

### Model Comparison
- **Higher bars** = Better performance
- **Small gap** between Test and CV = Good generalization
- **Large gap** = Possible overfitting

### Best Model Performance
- **Left panel**: Overall performance comparison
- **Right panel**: Consistency check
  - Tight box = Consistent performance
  - Wide box = Variable performance
  - Test score line should be within CV range

### Learning Curve
- **Converging lines** = Good learning
- **Gap between lines** = Overfitting (if training > validation)
- **Both low** = Underfitting
- **Both high and close** = Good fit

### Confusion Matrix
- **Diagonal values** = Correct predictions
- **Off-diagonal** = Errors
- Darker colors = Higher counts

### Metrics Comparison
- **All bars close to 1.0** = Excellent model
- **Balanced bars** = Good performance across metrics
- **Low recall** = Missing positive cases
- **Low precision** = Many false positives

## Customization

### Change Output Directory
Edit `ml_pipeline/advanced_pipeline.py`:

```python
self.visualizations_dir = Path("custom/path/visualizations")
```

### Modify Plot Styles
Edit `ml_pipeline/model_visualizer.py`:

```python
# Change colors
colors = ['#your_color1', '#your_color2']

# Change figure size
plt.rcParams['figure.figsize'] = (16, 10)

# Change DPI
plt.savefig(path, dpi=150)  # Lower DPI = smaller file
```

### Add Custom Visualizations
Extend `ModelVisualizer` class:

```python
def plot_custom(self, data):
    # Your custom plotting code
    pass
```

## Integration with CI/CD

Visualizations are automatically generated during training and can be:
- Saved as artifacts in CI/CD pipelines
- Included in reports
- Shared with stakeholders

## Troubleshooting

### Issue: No visualizations generated

**Check**:
1. Matplotlib backend: `import matplotlib; matplotlib.use('Agg')`
2. Directory permissions: `models/visualizations/` writable
3. Check logs for errors

### Issue: Plots are empty

**Solution**: Ensure model training completed successfully

### Issue: Learning curve fails

**Solution**: Some models (like VotingClassifier) don't support learning_curve
- Automatically handled - skips if not supported

## Best Practices

1. **Review all visualizations** after training
2. **Compare with previous runs** to track improvements
3. **Share visualizations** with team/stakeholders
4. **Include in reports** for documentation
5. **Version control** visualization outputs (optional)

## Example Output

After running `python train_model.py`, you'll see:

```
Creating visualizations...
Model comparison plot saved to models/visualizations/model_comparison.png
Best model performance plot saved to models/visualizations/best_model_performance.png
Learning curve plot saved to models/visualizations/learning_curve.png
Confusion matrix plot saved to models/visualizations/confusion_matrix.png
Metrics comparison plot saved to models/visualizations/metrics_comparison.png
All visualizations saved to models/visualizations
```

## References

- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [scikit-learn Learning Curves](https://scikit-learn.org/stable/modules/learning_curve.html)

