"""Breast-cancer classification, cross-validation, tuning, and reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _model_configurations(random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Return comparable pipelines and small, assignment-friendly parameter grids."""
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=3000,
                            random_state=random_state,
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", DecisionTreeClassifier(random_state=random_state)),
                ]
            ),
            {
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(random_state=random_state, n_jobs=-1),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "K-Nearest Neighbors": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            {
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"],
            },
        ),
    }


def run_classification(
    figures_dir: Path,
    reports_dir: Path,
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate four classifiers and tune them using training-only stratified CV."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target
    print(
        f"\nBreast Cancer Wisconsin dataset shape: {features.shape}; "
        f"classes: {dict(enumerate(dataset.target_names))}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model_configurations = _model_configurations(random_state)
    comparison_rows: list[dict[str, object]] = []
    tuned_details: dict[str, dict[str, object]] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for model_name, (pipeline, parameter_grid) in model_configurations.items():
        print(f"\nEvaluating {model_name}...")
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring="f1",
            n_jobs=-1,
            error_score="raise",
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1",
            cv=cross_validation,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_
        predictions = tuned_model.predict(X_test)
        probabilities = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = float(auc(fpr, tpr))
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=dataset.target_names,
        ).plot(cmap="Blues", values_format="d", ax=axis, colorbar=False)
        axis.set_title(f"{model_name}: Confusion Matrix")
        figure.tight_layout()
        figure.savefig(figures_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('-', '')}.png", dpi=160)
        plt.close(figure)

        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-Score": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc,
                "CV Mean": float(np.mean(cv_results["test_score"])),
                "CV Std": float(np.std(cv_results["test_score"])),
            }
        )
        roc_data[model_name] = (fpr, tpr, roc_auc)
        tuned_details[model_name] = {
            "best_parameters": search.best_params_,
            "best_cv_f1": float(search.best_score_),
            "confusion_matrix": matrix.tolist(),
        }

    results = pd.DataFrame(comparison_rows).sort_values("F1-Score", ascending=False)
    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    print("\nTuned test-set model comparison:\n", results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axis = plt.subplots(figsize=(9, 7))
    for model_name, (false_positive_rate, true_positive_rate, roc_auc) in roc_data.items():
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    axis.set(
        title="ROC Curve Comparison (Tuned Models)",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(figures_dir / "roc_curve_comparison.png", dpi=160)
    plt.close(figure)

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    chart_data = results.melt(id_vars="Model", value_vars=metric_names, var_name="Metric", value_name="Score")
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=chart_data, x="Model", y="Score", hue="Metric", ax=axis)
    axis.set(title="Tuned Classifier Performance on the Test Set", ylim=(0, 1.05), xlabel="", ylabel="Score")
    axis.tick_params(axis="x", rotation=12)
    axis.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(figures_dir / "model_performance_comparison.png", dpi=160)
    plt.close(figure)

    return {
        "results": results,
        "tuned_details": tuned_details,
        "dataset_shape": features.shape,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_names": dataset.target_names.tolist(),
        "best_model": str(results.iloc[0]["Model"]),
        "best_f1": float(results.iloc[0]["F1-Score"]),
    }


def write_final_report(
    report_path: Path,
    customer_source: str,
    customer_checks: dict[str, object],
    clustering: dict[str, object],
    classification: dict[str, object],
) -> None:
    """Write a student-friendly report using measured results from this run."""
    table = classification["results"]
    assert isinstance(table, pd.DataFrame)
    table_markdown = table.to_markdown(index=False, floatfmt=".3f")
    cluster_rows = "\n".join(
        f"| {row['clusters']} | {row['inertia']:.2f} | {row['silhouette']:.3f} |"
        for row in clustering["cluster_table"]
    )
    hierarchy_rows = "\n".join(
        f"| {row['clusters']} | {row['silhouette']:.3f} |"
        for row in clustering["hierarchical_cluster_table"]
    )
    tuning_lines = "\n".join(
        f"- **{name}:** best parameters `{details['best_parameters']}`; "
        f"best cross-validation F1 = {details['best_cv_f1']:.3f}."
        for name, details in classification["tuned_details"].items()
    )
    matrix_lines = "\n".join(
        f"- **{name}:** `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]` "
        "(rows are actual malignant/benign; columns are predicted malignant/benign)."
        for name, details in classification["tuned_details"].items()
        for matrix in [details["confusion_matrix"]]
    )
    report = f"""# Customer Segmentation and Machine Learning Model Evaluation

## 1. Introduction
This project explores customer groups without known labels, then evaluates four supervised classifiers on a separate medical dataset. The two tasks use different data and answer different questions.

## 2. Objectives
- Compare K-Means and agglomerative clustering for customer segmentation.
- Use PCA to summarize and visualize standardized customer features.
- Compare classification models using a stratified holdout set and five-fold cross-validation.
- Tune model hyperparameters and report test-set metrics.

## 3. Dataset Description
**Clustering:** {customer_source}. The loaded dataset has {clustering['records']} rows and uses age, annual income (thousands), and spending score. Customer ID and gender are not used as distance features. If the public CSV could not be reached, the saved offline fallback is synthetic, not observed customer data.

**Classification:** scikit-learn's Wisconsin Diagnostic Breast Cancer dataset, with {classification['dataset_shape'][0]} observations and {classification['dataset_shape'][1]} numeric input features. The target classes are malignant (0) and benign (1). No relationship between customer clusters and cancer labels is assumed.

## 4. Data Preprocessing
The customer data shape before cleaning was {customer_checks['shape']}. There were {customer_checks['duplicates']} duplicate records before duplicate removal. Numeric missing values are filled with their feature median. Features are standardized to mean zero and unit variance because income, age, and spending score use different units and ranges; distance-based clustering could otherwise be dominated by the largest-scale feature. Classification uses a stratified 80/20 split, median imputation, and scaling in pipelines where needed. Pipelines keep learned preprocessing within each training fold.

## 5. K-Means Clustering
K-Means assigns each customer to the nearest centroid and iteratively updates the centroids. The selected value was **K = {clustering['selected_k']}**, chosen as the tested K with the highest silhouette score. Cluster sizes (cluster: count) were `{clustering['cluster_sizes']}`. The income-versus-spending scatter plot offers a practical view of customer segments.

## 6. Elbow Method
Inertia measures the total squared distance from customers to their assigned centroids; it decreases as K grows. The elbow curve is saved in `outputs/figures/elbow_curve.png` and should be read alongside silhouette scores rather than treated as an exact automatic rule.

## 7. Silhouette Analysis
Silhouette scores summarize how close observations are to their own cluster compared with other clusters. Scores nearer 1 indicate clearer separation; values near 0 indicate overlap. Tested K-Means results:

| K | Inertia | Silhouette |
|---:|---:|---:|
{cluster_rows}

The selected K-Means silhouette score was **{clustering['kmeans_silhouette']:.3f}**.

## 8. Hierarchical Clustering
Agglomerative clustering repeatedly joins nearby groups, and Ward linkage is used for the dendrogram. The best tested cluster count by silhouette was **{clustering['hierarchical_k']}**, with score **{clustering['hierarchical_silhouette']:.3f}**. Its adjusted Rand index against K-Means was **{clustering['adjusted_rand_index']:.3f}**; this comparison is label-permutation invariant. A low agreement can mean the methods partition the same data differently, not that one is automatically wrong.

| Clusters | Hierarchical silhouette |
|---:|---:|
{hierarchy_rows}

## 9. PCA Dimensionality Reduction
PCA rotates standardized features into ordered directions that capture variance. It helps show multi-feature customer patterns on a two-dimensional plot; reducing dimensions may also simplify later workflows, though some information is lost. The first two components explain **{clustering['two_component_variance']:.1%}** of variance (individual ratios: `{[round(value, 4) for value in clustering['explained_variance_ratio']]}`). See the cumulative variance and 2D cluster figures.

## 10. Classification Models
Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors were trained on the training split. Test performance below uses each model's selected hyperparameters. ROC curves use predicted probability for the benign class.

## 11. Cross-Validation
Five-fold Stratified K-Fold was run on the training partition. Stratification preserves the approximate class balance in each fold. CV Mean and CV Std below are the mean and population standard deviation of the **baseline pipeline's F1** across those folds; hyperparameter search separately uses the same fold strategy and F1 scoring.

## 12. Confusion Matrix
Confusion matrices use malignant and benign in that order for both rows (actual) and columns (predicted). Diagonal counts are correct predictions; off-diagonal counts are mistakes.

{matrix_lines}

## 13. Precision, Recall and F1-score
Precision measures how often a predicted benign case is actually benign; recall measures how many actual benign cases are found (positive class = benign). F1 balances precision and recall. These measures complement accuracy when class errors have different consequences.

## 14. ROC-AUC
ROC curves show the true-positive rate against the false-positive rate over classification thresholds. ROC-AUC summarizes ranking performance; a value closer to 1 indicates better separation. Each tuned model's AUC is listed in the comparison table and plotted in `outputs/figures/roc_curve_comparison.png`.

## 15. Hyperparameter Tuning
GridSearchCV selected parameters using **F1-score** and only the training data. The best cross-validation score is the search's mean five-fold F1.

{tuning_lines}

## 16. Model Comparison

{table_markdown}

The table is also saved as `outputs/reports/model_comparison.csv`. The performance chart is saved as `outputs/figures/model_performance_comparison.png`.

## 17. Results and Observations
The top tuned model by test F1 in this run was **{classification['best_model']}** (F1 = {classification['best_f1']:.3f}). Test scores are estimates from a single stratified holdout and can vary with the split. Clustering metrics evaluate geometric separation, not a known correct customer label. PCA is a visualization aid and does not prove that segments have business value.

## 18. Conclusion
This end-to-end exercise demonstrates two unsupervised clustering methods, PCA, stratified cross-validation, standard classification metrics, ROC analysis, and grid-search tuning. The outputs are calculated from the loaded data each time the project runs.

## 19. Future Improvements
- Validate segments with business stakeholders and customer outcomes.
- Compare clustering on additional features and alternative distance/linkage choices.
- Use nested cross-validation for a less optimistic estimate after model selection.
- Evaluate probability calibration and choose decision thresholds for the use case.
- Add external validation data and monitor model performance over time.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")