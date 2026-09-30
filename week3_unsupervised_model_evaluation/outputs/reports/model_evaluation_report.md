# Customer Segmentation and Machine Learning Model Evaluation

## 1. Introduction
This project explores customer groups without known labels, then evaluates four supervised classifiers on a separate medical dataset. The two tasks use different data and answer different questions.

## 2. Objectives
- Compare K-Means and agglomerative clustering for customer segmentation.
- Use PCA to summarize and visualize standardized customer features.
- Compare classification models using a stratified holdout set and five-fold cross-validation.
- Tune model hyperparameters and report test-set metrics.

## 3. Dataset Description
**Clustering:** Public Mall Customers CSV (local copy). The loaded dataset has 200 rows and uses age, annual income (thousands), and spending score. Customer ID and gender are not used as distance features. If the public CSV could not be reached, the saved offline fallback is synthetic, not observed customer data.

**Classification:** scikit-learn's Wisconsin Diagnostic Breast Cancer dataset, with 569 observations and 30 numeric input features. The target classes are malignant (0) and benign (1). No relationship between customer clusters and cancer labels is assumed.

## 4. Data Preprocessing
The customer data shape before cleaning was (200, 5). There were 0 duplicate records before duplicate removal. Numeric missing values are filled with their feature median. Features are standardized to mean zero and unit variance because income, age, and spending score use different units and ranges; distance-based clustering could otherwise be dominated by the largest-scale feature. Classification uses a stratified 80/20 split, median imputation, and scaling in pipelines where needed. Pipelines keep learned preprocessing within each training fold.

## 5. K-Means Clustering
K-Means assigns each customer to the nearest centroid and iteratively updates the centroids. The selected value was **K = 6**, chosen as the tested K with the highest silhouette score. Cluster sizes (cluster: count) were `{0: 45, 1: 39, 2: 33, 3: 39, 4: 23, 5: 21}`. The income-versus-spending scatter plot offers a practical view of customer segments.

## 6. Elbow Method
Inertia measures the total squared distance from customers to their assigned centroids; it decreases as K grows. The elbow curve is saved in `outputs/figures/elbow_curve.png` and should be read alongside silhouette scores rather than treated as an exact automatic rule.

## 7. Silhouette Analysis
Silhouette scores summarize how close observations are to their own cluster compared with other clusters. Scores nearer 1 indicate clearer separation; values near 0 indicate overlap. Tested K-Means results:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 389.39 | 0.335 |
| 3 | 295.21 | 0.358 |
| 4 | 205.23 | 0.404 |
| 5 | 168.25 | 0.417 |
| 6 | 133.87 | 0.428 |
| 7 | 117.01 | 0.417 |
| 8 | 103.87 | 0.408 |
| 9 | 93.09 | 0.418 |
| 10 | 82.39 | 0.407 |

The selected K-Means silhouette score was **0.428**.

## 8. Hierarchical Clustering
Agglomerative clustering repeatedly joins nearby groups, and Ward linkage is used for the dendrogram. The best tested cluster count by silhouette was **6**, with score **0.420**. Its adjusted Rand index against K-Means was **0.891**; this comparison is label-permutation invariant. A low agreement can mean the methods partition the same data differently, not that one is automatically wrong.

| Clusters | Hierarchical silhouette |
|---:|---:|
| 2 | 0.318 |
| 3 | 0.321 |
| 4 | 0.361 |
| 5 | 0.390 |
| 6 | 0.420 |
| 7 | 0.398 |
| 8 | 0.366 |
| 9 | 0.375 |
| 10 | 0.381 |

## 9. PCA Dimensionality Reduction
PCA rotates standardized features into ordered directions that capture variance. It helps show multi-feature customer patterns on a two-dimensional plot; reducing dimensions may also simplify later workflows, though some information is lost. The first two components explain **77.6%** of variance (individual ratios: `[0.4427, 0.3331]`). See the cumulative variance and 2D cluster figures.

## 10. Classification Models
Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors were trained on the training split. Test performance below uses each model's selected hyperparameters. ROC curves use predicted probability for the benign class.

## 11. Cross-Validation
Five-fold Stratified K-Fold was run on the training partition. Stratification preserves the approximate class balance in each fold. CV Mean and CV Std below are the mean and population standard deviation of the **baseline pipeline's F1** across those folds; hyperparameter search separately uses the same fold strategy and F1 scoring.

## 12. Confusion Matrix
Confusion matrices use malignant and benign in that order for both rows (actual) and columns (predicted). Diagonal counts are correct predictions; off-diagonal counts are mistakes.

- **Logistic Regression:** `[[41, 1], [1, 71]]` (rows are actual malignant/benign; columns are predicted malignant/benign).
- **Decision Tree:** `[[39, 3], [6, 66]]` (rows are actual malignant/benign; columns are predicted malignant/benign).
- **Random Forest:** `[[39, 3], [2, 70]]` (rows are actual malignant/benign; columns are predicted malignant/benign).
- **K-Nearest Neighbors:** `[[39, 3], [0, 72]]` (rows are actual malignant/benign; columns are predicted malignant/benign).

## 13. Precision, Recall and F1-score
Precision measures how often a predicted benign case is actually benign; recall measures how many actual benign cases are found (positive class = benign). F1 balances precision and recall. These measures complement accuracy when class errors have different consequences.

## 14. ROC-AUC
ROC curves show the true-positive rate against the false-positive rate over classification thresholds. ROC-AUC summarizes ranking performance; a value closer to 1 indicates better separation. Each tuned model's AUC is listed in the comparison table and plotted in `outputs/figures/roc_curve_comparison.png`.

## 15. Hyperparameter Tuning
GridSearchCV selected parameters using **F1-score** and only the training data. The best cross-validation score is the search's mean five-fold F1.

- **Logistic Regression:** best parameters `{'classifier__C': 0.1}`; best cross-validation F1 = 0.986.
- **Decision Tree:** best parameters `{'classifier__max_depth': 5, 'classifier__min_samples_split': 2}`; best cross-validation F1 = 0.937.
- **Random Forest:** best parameters `{'classifier__max_depth': None, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 200}`; best cross-validation F1 = 0.970.
- **K-Nearest Neighbors:** best parameters `{'classifier__n_neighbors': 9, 'classifier__weights': 'uniform'}`; best cross-validation F1 = 0.972.

## 16. Model Comparison

| Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   CV Mean |   CV Std |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|----------:|---------:|
| Logistic Regression |      0.982 |       0.986 |    0.986 |      0.986 |     0.996 |     0.984 |    0.007 |
| K-Nearest Neighbors |      0.974 |       0.960 |    1.000 |      0.980 |     0.994 |     0.970 |    0.009 |
| Random Forest       |      0.956 |       0.959 |    0.972 |      0.966 |     0.993 |     0.970 |    0.015 |
| Decision Tree       |      0.921 |       0.957 |    0.917 |      0.936 |     0.916 |     0.932 |    0.014 |

The table is also saved as `outputs/reports/model_comparison.csv`. The performance chart is saved as `outputs/figures/model_performance_comparison.png`.

## 17. Results and Observations
The top tuned model by test F1 in this run was **Logistic Regression** (F1 = 0.986). Test scores are estimates from a single stratified holdout and can vary with the split. Clustering metrics evaluate geometric separation, not a known correct customer label. PCA is a visualization aid and does not prove that segments have business value.

## 18. Conclusion
This end-to-end exercise demonstrates two unsupervised clustering methods, PCA, stratified cross-validation, standard classification metrics, ROC analysis, and grid-search tuning. The outputs are calculated from the loaded data each time the project runs.

## 19. Future Improvements
- Validate segments with business stakeholders and customer outcomes.
- Compare clustering on additional features and alternative distance/linkage choices.
- Use nested cross-validation for a less optimistic estimate after model selection.
- Evaluate probability calibration and choose decision thresholds for the use case.
- Add external validation data and monitor model performance over time.
