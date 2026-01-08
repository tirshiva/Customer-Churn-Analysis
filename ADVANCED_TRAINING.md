# Advanced Model Training Guide

## Overview

The project now includes an **advanced training pipeline** that:
- Trains **10 different classification models**
- Uses **GridSearchCV** for hyperparameter tuning
- Applies **feature scaling** and **SMOTE** for class balancing
- Creates **ensemble models** from top performers
- Automatically selects the **best performing model**

## Models Trained

1. **Random Forest** - Ensemble of decision trees
2. **Gradient Boosting** - Sequential boosting
3. **AdaBoost** - Adaptive boosting
4. **Extra Trees** - Extremely randomized trees
5. **Logistic Regression** - Linear classifier
6. **SVM** - Support Vector Machine
7. **K-Nearest Neighbors** - Instance-based learning
8. **Naive Bayes** - Probabilistic classifier
9. **Decision Tree** - Single tree classifier
10. **Neural Network** - Multi-layer perceptron

## Performance Improvements

### Techniques Applied

1. **Feature Scaling** (StandardScaler)
   - Normalizes features to have zero mean and unit variance
   - Improves performance for distance-based and neural network models

2. **SMOTE (Synthetic Minority Oversampling)**
   - Balances class distribution
   - Reduces bias towards majority class
   - Improves recall for minority class

3. **GridSearchCV**
   - Exhaustive search over hyperparameter space
   - 5-fold cross-validation
   - Finds optimal hyperparameters for each model

4. **Dimension Reduction**
   - Selects most impactful features
   - Reduces overfitting
   - Speeds up training

5. **Ensemble Methods**
   - Combines top 3 models
   - Voting classifier with soft voting
   - Often outperforms individual models

## Usage

### Basic Training

```bash
python train_model.py
```

This will:
1. Load and preprocess data
2. Apply dimension reduction
3. Scale features
4. Balance classes with SMOTE
5. Train all 10 models with GridSearchCV
6. Create ensemble model
7. Select and save best model
8. Generate comparison report

### Expected Output

```
======================================================================
STARTING ADVANCED ML PIPELINE
======================================================================

[1/6] Data Ingestion
Data loaded successfully. Shape: (7043, 21)

[2/6] Data Preprocessing
Preprocessing completed. Shape: (7043, 46)

[3/6] Preparing Training Data
Scaling features...
Applying SMOTE for class balancing...

[4/6] Training All Models with GridSearchCV
Training RANDOM_FOREST...
Training GRADIENT_BOOSTING...
...

[5/6] Creating Ensemble Model
Creating ensemble from top 3 models...

[6/6] Evaluating Best Model
Best Model: gradient_boosting
Test Accuracy: 0.8523
```

## Model Comparison

After training, check `models/model_comparison.json`:

```json
{
  "best_model": "gradient_boosting",
  "ranking": ["gradient_boosting", "random_forest", "ensemble", ...],
  "models": {
    "gradient_boosting": {
      "test_accuracy": 0.8523,
      "cv_mean_accuracy": 0.8489,
      "best_parameters": {...}
    },
    ...
  }
}
```

## Hyperparameter Grids

### Random Forest
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]

### Gradient Boosting
- `n_estimators`: [100, 200]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 5, 7]
- `min_samples_split`: [2, 5]
- `subsample`: [0.8, 1.0]

### Logistic Regression
- `C`: [0.001, 0.01, 0.1, 1, 10, 100]
- `penalty`: ['l1', 'l2', 'elasticnet']
- `solver`: ['liblinear', 'lbfgs', 'saga']

### And more for each model...

## Configuration Options

Edit `train_model.py` to customize:

```python
pipeline = AdvancedMLPipeline(
    ...
    use_scaling=True,      # Enable/disable feature scaling
    use_smote=True,        # Enable/disable SMOTE
    create_ensemble=True,  # Create ensemble model
    dimension_reduction_method="select_from_model",
    dimension_reduction_k=None
)
```

## Performance Expectations

### Before (Single Random Forest)
- Accuracy: ~79%
- No hyperparameter tuning
- No feature scaling
- No class balancing

### After (Advanced Pipeline)
- Accuracy: **82-85%** (expected improvement)
- All models tuned with GridSearchCV
- Feature scaling applied
- Class balancing with SMOTE
- Best model automatically selected

## Training Time

- **Single Model**: ~1-2 minutes
- **All Models**: ~10-30 minutes (depending on hardware)
- **With GridSearchCV**: Longer but finds optimal parameters

## Output Files

After training:

1. **`models/best_model.joblib`**
   - Best performing model
   - Includes model, scaler, and metadata

2. **`models/evaluation_report.json`**
   - Detailed evaluation metrics
   - Confusion matrix
   - Classification report

3. **`models/model_comparison.json`**
   - Comparison of all models
   - Test accuracy for each
   - Best hyperparameters

4. **`models/dimension_reducer.joblib`**
   - Dimension reducer (if used)

## Troubleshooting

### Issue: Training takes too long

**Solution**: Reduce hyperparameter grid size in `model_trainer.py`:

```python
'params': {
    'n_estimators': [100, 200],  # Reduce options
    'max_depth': [10, 20],        # Fewer values
}
```

### Issue: SMOTE fails

**Solution**: Automatically handled - falls back to no SMOTE if insufficient samples

### Issue: Out of memory

**Solution**: 
- Reduce `n_jobs` in GridSearchCV
- Use smaller hyperparameter grids
- Disable ensemble creation

### Issue: Low accuracy

**Solutions**:
1. Check data quality
2. Try different dimension reduction methods
3. Adjust SMOTE parameters
4. Increase hyperparameter grid size
5. Try different feature engineering

## Best Practices

1. **Always compare models**: Check `model_comparison.json`
2. **Review best parameters**: Learn what works best
3. **Monitor training time**: Balance performance vs. time
4. **Validate on test set**: Don't overfit to validation
5. **Save all artifacts**: Models, reducers, reports

## Advanced Customization

### Custom Hyperparameter Grids

Edit `ml_pipeline/model_trainer.py`:

```python
def get_model_configs(self):
    configs = {
        'random_forest': {
            'model': RandomForestClassifier(...),
            'params': {
                # Your custom grid
                'n_estimators': [50, 100, 150],
                ...
            }
        },
        ...
    }
```

### Add New Models

Add to `get_model_configs()`:

```python
'your_model': {
    'model': YourClassifier(),
    'params': {
        'param1': [val1, val2],
        ...
    }
}
```

## Performance Tips

1. **Use more CPU cores**: Set `n_jobs=-1`
2. **Reduce CV folds**: Use 3 instead of 5 for faster training
3. **Smaller grids**: Focus on most important parameters
4. **Early stopping**: Some models support early stopping
5. **Parallel processing**: GridSearchCV uses parallel by default

## Expected Improvements

With all techniques applied:
- **Accuracy**: 79% → **82-85%**
- **Precision**: Improved
- **Recall**: Improved (especially minority class)
- **F1 Score**: Improved
- **ROC AUC**: Improved

## Next Steps

1. Train models: `python train_model.py`
2. Review comparison: `cat models/model_comparison.json`
3. Check evaluation: `cat models/evaluation_report.json`
4. Use best model: Automatically loaded by prediction service

## References

- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)
- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Feature Scaling](https://scikit-learn.org/stable/modules/preprocessing.html)

