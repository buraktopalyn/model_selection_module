# Model Selection Module 🚀

This module is a tool that automatically evaluates machine learning models and selects the best performing model. It includes various algorithms for regression, classification, and clustering problems. Additionally, it provides comprehensive functions for data preprocessing and exploratory data analysis (EDA).

## Features 🌟

### Model Selector (ModelSelector)

- **Multiple Problem Type Support**: Can be used for regression, classification, and clustering problems
- **Wide Range of Models**: 
  - 14 regression algorithms
  - 14 classification algorithms
  - 5 clustering algorithms
- **Ensemble Models**: Bagging, stacking, boosting, and voting methods
- **Automatic Model Evaluation**: Trains all models and compares their performance
- **Best Model Selection**: Automatically selects the best model based on performance metrics

### Data Preprocessing (DataPreprocessor) 🛠️

- **Missing Value Handling**: Mean, median, mode, constant value, KNN, and row deletion methods
- **Outlier Detection and Handling**: Z-score, IQR, isolation forest methods
- **Feature Scaling**: Standard, MinMax, Robust, and Power transformations
- **Categorical Variable Encoding**: One-hot, Label, and Ordinal encoding
- **Feature Selection**: K-best, importance-based, RFE, and PCA methods
- **Exploratory Data Analysis (EDA)**: Visualization tools and statistical analyses

## Installation 🔧�

```bash
pip install -r requirements.txt
```

## Usage 📊

### Regression Example

```python
from model_selector import ModelSelector
import numpy as np

# Create data
X = np.random.rand(100, 5)
y = np.random.rand(100) * 10

# Create model selector
ms = ModelSelector(problem_type='regression')

# Fit models and evaluate
ms.fit(X, y)

# Display results
ms.display_results()

# Get best model
best_model = ms.get_best_model()
print(f"En iyi model: {ms.best_model_name}")
```

### Classification Example

```python
from model_selector import ModelSelector
import numpy as np

# Create data
X = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)  # 3-class classification

# Create model selector
ms = ModelSelector(problem_type='classification')

# Fit models and evaluate
ms.fit(X, y)

# Display results
ms.display_results()

# Get best model
best_model = ms.get_best_model()
print(f"En iyi model: {ms.best_model_name}")
```

### Clustering Example

```python
from model_selector import ModelSelector
import numpy as np

# Create data
X = np.random.rand(100, 5)

# Create model selector
ms = ModelSelector(problem_type='clustering')

# Fit models and evaluate
ms.fit(X, None)

# Display results
ms.display_results()
```

### Data Preprocessing Example

```python
from data_preprocessor import DataPreprocessor
import pandas as pd

# Create data
data = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [10, 20, 30, None, 50],
    'C': ['x', 'y', 'z', 'x', 'y']
})

# Define preprocessing steps
preprocessing_steps = {
    'handle_missing_values': {'method': 'mean'},
    'encode_categorical': {'method': 'one_hot'},
    'scale_features': {'method': 'standard'}
}

# Create data preprocessor
preprocessor = DataPreprocessor(verbose=True)

# Preprocess data
processed_data = preprocessor.fit_transform(
    data=data,
    preprocessing_steps=preprocessing_steps
)

print(processed_data.head())
```

### Ensemble Model Adding Example

```python
from model_selector import ModelSelector
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Create data
X = np.random.rand(100, 5)
y = np.random.rand(100) * 10

# Create model selector
ms = ModelSelector(problem_type='regression')

# Add ensemble models
base_models = [Ridge(), Lasso(), RandomForestRegressor()]
ms.add_ensemble_model('stacking', base_models=base_models)

# Fit models and evaluate
ms.fit(X, y)
```

### Full Example Usage

For a more comprehensive example, you can check the `example_usage.py` file. This file demonstrates how to use data preprocessing and model selection modules together.

## Test Files 🧪

The project includes test files for different problem types:

- **test_regression.py**: Test file for regression models (California Housing dataset)
- **test_classification.py**: Test file for classification models (Iris dataset)
- **test_clustering.py**: Test file for clustering models (synthetic dataset)

For running the test files:

```bash
python test_regression.py
python test_classification.py
python test_clustering.py
```

## Supported Models

### Regression Models
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting
- SVR
- KNN
- AdaBoost
- Bagging
- XGBoost
- LightGBM
- CatBoost

### Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVC
- KNN
- Naive Bayes
- LDA
- QDA
- AdaBoost
- Bagging
- XGBoost
- LightGBM
- CatBoost

### Clustering Models
- KMeans
- Agglomerative
- DBSCAN
- Spectral
- Birch

## Licence 📜
This project is licensed under the MIT License.
