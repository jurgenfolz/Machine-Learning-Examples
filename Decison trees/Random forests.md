Random Forests
==============

**Random Forest** is an ensemble learning method used for both classification and regression tasks. It builds multiple decision trees during training and combines their predictions to improve accuracy and reduce overfitting. Random forests enhance the robustness of individual decision trees by introducing randomness in both feature selection and data sampling.

1. Model Definition
-------------------

A random forest consists of multiple decision trees trained on different subsets of data. The final prediction is obtained by aggregating the individual tree predictions.

### Steps to Build a Random Forest:
1. Draw **bootstrap samples** from the original dataset (random sampling with replacement).
2. Train each decision tree on its corresponding bootstrap sample.
3. At each node, a **random subset of features** is selected to determine the best split (instead of considering all features).
4. Aggregate the predictions:
   - **Classification**: Majority voting across trees.
   - **Regression**: Averaging the output of individual trees.

2. Cost Function
----------------

Random forests use decision trees as base models. Each tree is trained using the **Gini impurity**, **entropy**, or **mean squared error (MSE)** as the cost function, depending on the task:

- **Classification**: 
 $`
  Gini = 1 - \sum_{i=1}^{k} p_i^2
 `$
  or  
 $`
  Entropy = - \sum_{i=1}^{k} p_i \log_2(p_i)
 `$
  where \( p_i \) is the proportion of instances in class \( i \).

- **Regression**:
 $`
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
 `$

3. Parameter Estimation
-----------------------

The key hyperparameters for training a random forest include:

- **n_estimators**: The number of trees in the forest.
- **max_features**: The number of features randomly chosen at each split.
- **max_depth**: The maximum depth of each decision tree.
- **min_samples_split**: The minimum number of samples required to split a node.
- **min_samples_leaf**: The minimum number of samples required at a leaf node.
- **bootstrap**: Whether to use bootstrap samples for training.

The optimal values for these parameters can be determined via hyperparameter tuning, such as grid search or random search.

4. Interpretation of Predictions
--------------------------------

For classification, the final prediction is determined by majority voting:

$`
\hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_T)
`$

For regression, the final prediction is the average of the individual tree predictions:

$`
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t
`$

where \( T \) is the total number of trees.

5. Assumptions
--------------

Random forests make fewer assumptions than traditional models, but key considerations include:

1. **Independence of Observations**: Data points are assumed to be independent.
2. **Sufficient Training Data**: Requires a large dataset to perform well.
3. **Feature Relevance**: Works best when relevant features are included.

6. Performance Evaluation
-------------------------

The performance of a random forest is evaluated using different metrics depending on the task:

- **Classification**: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- **Regression**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), \( R^2 \) score.

7. Extensions
-------------

- **Extra Trees (Extremely Randomized Trees)**: Introduces additional randomness by selecting split thresholds randomly.
- **Gradient Boosted Trees**: Uses boosting instead of bagging to improve performance.
- **Feature Importance Analysis**: Measures how important each feature is for prediction.

8. Pros and Cons
----------------

### Pros:
1. **Reduces Overfitting**: Averages multiple trees, preventing overfitting.
2. **Handles Missing Data**: Can handle missing values without imputation.
3. **Works Well with High-Dimensional Data**: Can model complex relationships.
4. **Feature Importance**: Identifies significant features for predictions.
5. **Robust to Noisy Data**: More stable compared to a single decision tree.

### Cons:
1. **Computationally Expensive**: Training multiple trees requires significant resources.
2. **Less Interpretability**: Harder to interpret than a single decision tree.
3. **Memory Intensive**: Storing multiple trees requires more memory.
4. **Slower Predictions**: Making predictions can be slower than simpler models.

Random forests are widely used in fields such as finance, healthcare, and image recognition due to their high accuracy and robustness. They form the basis for more advanced ensemble techniques like boosting.
