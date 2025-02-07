Linear Regression (Ordinary Least Squares)
==========================================

**Linear Regression**, also known as **Ordinary Least Squares (OLS)**, is a fundamental supervised learning algorithm used for regression tasks. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the data.

1.  Model Definition

* * * * *

Given a dataset D={(xi,yi)}D = \{(x_i, y_i)\} where xix_i is a feature vector and yiy_i is the target variable, linear regression assumes a linear relationship:

$`dy=β0+β1x1+β2x2+⋯+βdxd+ϵy = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_d x_d + \epsilon`$

where:

-   $`β0\beta_0`$ is the intercept (bias term),
-   $`β1,β2,...,βd\beta_1, \beta_2, ..., \beta_d`$ are the regression coefficients (weights),
-   $`x1,x2,...,xdx_1, x_2, ..., x_d`$ are the feature values,
-   $`\epsilon`$ is the error term, representing noise or unmodeled variation in the data.

1.  Cost Function

* * * * *

The **Ordinary Least Squares (OLS)** method estimates the parameters β\beta by minimizing the sum of squared residuals:

$`J(β)=∑i=1n(yi-y^i)2J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2`$

where $`y^i\hat{y}_i`$ is the predicted value given by:

$`y^=Xβ\hat{y} = X \beta`$

1.  Parameter Estimation

* * * * *

The optimal values for β\beta can be obtained using the **normal equation**:

$`β=(XTX)-1XTy\beta = (X^T X)^{-1} X^T y`$

where:

-   $`XX`$ is the design matrix containing the feature values,
-   $`yy`$ is the target variable vector.

For large datasets, iterative methods like **Gradient Descent** are preferred over the normal equation due to computational efficiency.

1.  Assumptions

* * * * *

Linear regression relies on the following assumptions:

1.  **Linearity**: The relationship between the independent and dependent variables is linear.

2.  **Independence**: The residuals (errors) are independent of each other.

3.  **Homoscedasticity**: The variance of residuals is constant across all levels of the independent variables.

4.  **Normality**: The residuals follow a normal distribution.

5.  **No multicollinearity**: Independent variables should not be highly correlated with each other.

6.  Performance Evaluation

* * * * *

The performance of a linear regression model is typically evaluated using:

-   **Mean Squared Error (MSE)**:

    $`MSE=1n∑i=1n(yi-y^i)2MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2`$

-   **Root Mean Squared Error (RMSE)**:

    $`RMSE=MSERMSE = \sqrt{MSE}`$

-   **R-squared (R2R^2) Score**:

    $`R2=1-∑(yi-y^i)2∑(yi-yˉ)2R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}`$

    where $`yˉ\bar{y}`$ is the mean of yy.

1.  Extensions

* * * * *

-   **Multiple Linear Regression**: Extends simple linear regression to multiple predictors.
-   **Polynomial Regression**: Models non-linear relationships by introducing polynomial terms.
-   **Ridge Regression (L2 Regularization)**: Adds a penalty term $`λ∑β2\lambda \sum \beta^2`$ to prevent overfitting.
-   **Lasso Regression (L1 Regularization)**: Adds a penalty term $`λ∑∣β∣\lambda \sum |\beta|`$ to enforce sparsity.

1.  Pros and Cons

* * * * *

### Pros:

1.  **Simple and Interpretable**: Provides clear insights into feature relationships.
2.  **Efficient**: Computationally inexpensive for small to medium datasets.
3.  **Widely Used**: Forms the foundation of many advanced statistical and machine learning models.
4.  **Handles Continuous Data Well**: Works well for numerical target variables.

### Cons:

1.  **Sensitive to Outliers**: Can be heavily influenced by extreme values.
2.  **Assumptions Must Hold**: Performance degrades if assumptions like linearity and independence are violated.
3.  **Limited to Linear Relationships**: Cannot capture complex non-linear patterns without modifications.
4.  **Multicollinearity Issues**: Highly correlated independent variables can distort coefficient estimates.

Linear regression is a fundamental technique in machine learning and statistics, serving as a baseline model for many regression tasks. It is widely used in predictive modeling, finance, economics, and various engineering applications.