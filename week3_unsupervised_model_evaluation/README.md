# Customer Segmentation and Machine Learning Model Evaluation

An end-to-end Week 3 machine-learning assignment covering unsupervised customer segmentation and a separate supervised classification evaluation. Running one command downloads the public Mall Customers CSV (when online), fits clustering models, evaluates and tunes four classifiers on scikit-learn's Wisconsin breast-cancer dataset, saves figures and tables, and writes a results-driven Markdown report.

## Objectives

- Explore and prepare customer data, inspect missing values and duplicates, and standardize numeric features.
- Compare K-Means and agglomerative clustering with inertia, silhouette scores, cluster sizes, and a dendrogram.
- Apply PCA and visualize customer segments in two dimensions.
- Train Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors classifiers.
- Report accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and five-fold stratified CV.
- Tune model hyperparameters with GridSearchCV using F1-score.
- Save reproducible visualizations, result tables, cluster assignments, and a plain-language report.

## Technologies

Python 3.10+, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, SciPy, and Jupyter.

## Dataset Information

**Clustering:** The public [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) is downloaded from a [public CSV mirror](https://raw.githubusercontent.com/devotuoma/Mall-Customers-Segmentation/main/Mall_Customers.csv) into `data/Mall_Customers.csv`. Features are age, annual income (k$), and spending score. Customer ID and gender are excluded from distance calculations. When the CSV cannot be reached and is not already local, the program saves and labels a deterministic synthetic fallback; that fallback is not the public observed dataset.

**Classification:** The Wisconsin Diagnostic Breast Cancer dataset is loaded from `sklearn.datasets`. It is a distinct task and is not combined with the customer data.

## Project Structure

```text
week3_unsupervised_model_evaluation/
├── data/
│   └── README.md
├── notebooks/
│   └── week3_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── preprocessing.py
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── README.md
└── main.py
```

## Installation and Run

From PowerShell, in the folder containing this README:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run the environment's Python directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

To use a manually downloaded Mall Customers CSV, place it at `data/Mall_Customers.csv` before running. Jupyter is optional; install it separately with `python -m pip install jupyter` if you want to execute the included notebook, then run `jupyter notebook notebooks/week3_analysis.ipynb`.

## Expected Outputs

- `outputs/figures/`: elbow curve, silhouette scores, hierarchical dendrogram, PCA explained variance, K-Means clusters, PCA 2D clusters, four confusion matrices, ROC comparison, and model performance comparison.
- `outputs/reports/model_evaluation_report.md`: run-specific analysis and model results.
- `outputs/reports/model_comparison.csv`: test metrics and baseline cross-validation summary.
- `outputs/reports/customer_cluster_assignments.csv`: customer rows with their K-Means labels.
- `data/Mall_Customers.csv` or an explicitly labeled offline fallback CSV.

## Techniques and Metrics

K-Means, agglomerative clustering (Ward linkage), silhouette analysis, inertia/elbow analysis, PCA, StandardScaler, stratified train/test splitting, five-fold StratifiedKFold cross-validation, GridSearchCV, and Logistic Regression, Decision Tree, Random Forest, and KNN. Classification metrics are accuracy, precision, recall, F1-score, and ROC-AUC; confusion matrices and ROC curves are also saved.

## Conclusion

The project demonstrates a complete reproducible workflow for unsupervised segmentation and supervised model evaluation. Clustering quality scores describe separation rather than ground-truth segment correctness; classifier test scores are estimates from one stratified holdout. See the generated report for measured results and caveats.