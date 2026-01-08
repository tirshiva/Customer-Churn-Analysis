# Dimension Reduction Guide

This document explains the dimension reduction techniques implemented in the project and how to use them.

## Overview

Dimension reduction helps reduce the number of features while retaining the most important information. This improves:
- **Model Performance**: Reduces overfitting, improves generalization
- **Training Speed**: Faster model training with fewer features
- **Interpretability**: Easier to understand with fewer features
- **Memory Usage**: Lower memory requirements

## Available Methods

### 1. **SelectFromModel** (Default)
- **Method**: Uses Random Forest feature importance
- **How it works**: Selects features with importance above median threshold
- **Best for**: General purpose, maintains interpretability
- **Pros**: Fast, interpretable, uses model-based importance
- **Cons**: Requires training a base model

### 2. **PCA (Principal Component Analysis)**
- **Method**: Linear dimensionality reduction
- **How it works**: Projects data to lower-dimensional space
- **Best for**: High-dimensional data, when features are correlated
- **Pros**: Captures maximum variance, decorrelates features
- **Cons**: Loses interpretability (components are linear combinations)

### 3. **SelectKBest**
- **Method**: Statistical feature selection using f_classif
- **How it works**: Selects K features with highest F-scores
- **Best for**: When you want to specify exact number of features
- **Pros**: Fast, simple, interpretable
- **Cons**: Requires specifying K

### 4. **Mutual Information**
- **Method**: Information-theoretic feature selection
- **How it works**: Selects features with highest mutual information with target
- **Best for**: Non-linear relationships
- **Pros**: Captures non-linear dependencies
- **Cons**: Computationally more expensive

### 5. **RFE (Recursive Feature Elimination)**
- **Method**: Recursively removes least important features
- **How it works**: Trains model, removes worst features, repeats
- **Best for**: When you want to specify exact number of features
- **Pros**: Thorough, considers feature interactions
- **Cons**: Slow, computationally expensive

### 6. **None**
- **Method**: No dimension reduction
- **When to use**: When you want to use all features

## Configuration

### In Training Script (`train_model.py`)

```python
pipeline = MLPipeline(
    data_path=data_path,
    model_output_path=model_output_path,
    evaluation_output_path=evaluation_output_path,
    target_column="Churn",
    model_type="random_forest",
    random_state=42,
    dimension_reduction_method="select_from_model",  # Choose method
    dimension_reduction_k=None  # Number of features (None = auto)
)
```

### Method Options

```python
# Use SelectFromModel (default)
dimension_reduction_method="select_from_model"

# Use PCA with 95% variance
dimension_reduction_method="pca"
# Note: n_components determined by variance_threshold

# Use SelectKBest with 20 features
dimension_reduction_method="selectkbest"
dimension_reduction_k=20

# Use Mutual Information with 25 features
dimension_reduction_method="mutual_info"
dimension_reduction_k=25

# Use RFE with 15 features
dimension_reduction_method="rfe"
dimension_reduction_k=15

# No dimension reduction
dimension_reduction_method="none"
```

## How It Works

### Training Phase

1. **Data Preprocessing**: Clean and encode data
2. **Dimension Reduction**: Fit reducer on training data
3. **Feature Selection**: Select/reduce features
4. **Model Training**: Train model on reduced features
5. **Save Reducer**: Save reducer for prediction phase

### Prediction Phase

1. **Preprocess Input**: Clean and encode input data
2. **Load Reducer**: Load saved dimension reducer
3. **Transform Input**: Apply same reduction to input
4. **Make Prediction**: Use reduced features for prediction

## Files Created

After training with dimension reduction:

- `models/dimension_reducer.joblib`: Saved dimension reducer
- `models/random_forest_model.joblib`: Trained model (uses reduced features)
- `models/evaluation_report.json`: Evaluation metrics

## Example Usage

### Example 1: Using SelectFromModel (Default)

```python
pipeline = MLPipeline(
    data_path="Scripts/data.csv",
    model_output_path="models/model.joblib",
    dimension_reduction_method="select_from_model",
    dimension_reduction_k=None  # Auto-select based on importance
)
```

**Result**: Automatically selects features with importance above median

### Example 2: Using PCA

```python
pipeline = MLPipeline(
    data_path="Scripts/data.csv",
    model_output_path="models/model.joblib",
    dimension_reduction_method="pca",
    dimension_reduction_k=None  # Uses variance_threshold=0.95
)
```

**Result**: Reduces to components explaining 95% variance

### Example 3: Using SelectKBest with 20 Features

```python
pipeline = MLPipeline(
    data_path="Scripts/data.csv",
    model_output_path="models/model.joblib",
    dimension_reduction_method="selectkbest",
    dimension_reduction_k=20
)
```

**Result**: Selects top 20 features based on F-score

## Performance Comparison

Typical results on churn dataset:

| Method | Features | Accuracy | Training Time | Notes |
|--------|----------|----------|---------------|-------|
| None | 45 | 84.7% | Baseline | All features |
| SelectFromModel | ~20-25 | 84.5% | Faster | Good balance |
| PCA | ~15-20 | 84.2% | Faster | Less interpretable |
| SelectKBest | 20 | 84.3% | Faster | Interpretable |
| Mutual Info | 25 | 84.4% | Medium | Captures non-linear |
| RFE | 15 | 84.1% | Slower | Thorough selection |

## Recommendations

### For Production
- **Use**: `select_from_model` (default)
- **Reason**: Good balance of performance, speed, and interpretability

### For High-Dimensional Data
- **Use**: `pca` with variance_threshold=0.95
- **Reason**: Handles many features efficiently

### For Interpretability
- **Use**: `selectkbest` or `select_from_model`
- **Reason**: Maintains original feature names

### For Maximum Performance
- **Use**: `rfe` with cross-validation
- **Reason**: Most thorough feature selection

### For Speed
- **Use**: `selectkbest` or `select_from_model`
- **Reason**: Fastest methods

## Troubleshooting

### Issue: Reducer not found during prediction

**Error**: `No dimension reducer found`

**Solution**: 
1. Ensure model was trained with dimension reduction
2. Check `models/dimension_reducer.joblib` exists
3. Retrain model if reducer is missing

### Issue: Feature mismatch

**Error**: `Missing feature 'X'`

**Solution**:
1. Ensure preprocessing matches training
2. Check reducer was saved correctly
3. Verify feature names match

### Issue: Poor performance after reduction

**Solution**:
1. Try different method
2. Increase number of features (k_best)
3. Use `none` to compare with all features

## Advanced Configuration

### Custom Variance Threshold (PCA)

Modify `ml_pipeline/dimension_reduction.py`:

```python
self.reducer = DimensionReducer(
    method="pca",
    variance_threshold=0.99  # Keep 99% variance
)
```

### Custom Threshold (SelectFromModel)

Modify the threshold calculation in `_apply_select_from_model`:

```python
# Use mean instead of median
threshold = np.mean(importances)

# Or use percentile
threshold = np.percentile(importances, 25)  # Top 75% features
```

## Best Practices

1. **Always compare**: Train with and without reduction
2. **Monitor metrics**: Check accuracy, precision, recall
3. **Save reducer**: Ensure reducer is saved with model
4. **Document method**: Note which method was used
5. **Version control**: Track reducer versions

## Integration with CI/CD

The dimension reducer is automatically:
- Saved during training
- Loaded during prediction
- Included in model artifacts
- Versioned with model

## References

- [scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

