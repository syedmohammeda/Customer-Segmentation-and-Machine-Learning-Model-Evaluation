"""K-Means, agglomerative clustering, PCA, and their visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.preprocessing import prepare_customer_features


def run_clustering(
    customers: pd.DataFrame,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, object]:
    """Fit clustering models, save required figures, and return report values."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    customer_data, features, _ = prepare_customer_features(customers)
    cluster_counts = list(range(2, min(10, len(customer_data) - 1) + 1))

    kmeans_results: list[dict[str, float | int]] = []
    for cluster_count in cluster_counts:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        kmeans_results.append(
            {
                "clusters": cluster_count,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(features, labels)),
            }
        )
    metrics = pd.DataFrame(kmeans_results)
    selected_k = int(metrics.loc[metrics["silhouette"].idxmax(), "clusters"])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(metrics["clusters"], metrics["inertia"], marker="o")
    axes[0].set(title="K-Means Elbow Curve", xlabel="Number of clusters (K)", ylabel="Inertia")
    axes[1].plot(metrics["clusters"], metrics["silhouette"], marker="o", color="teal")
    axes[1].set(
        title="K-Means Silhouette Analysis",
        xlabel="Number of clusters (K)",
        ylabel="Silhouette score",
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "elbow_and_silhouette.png", dpi=160)
    plt.close(figure)

    # Save separate files as requested while retaining the two-panel overview.
    for column, title, ylabel, filename, color in [
        ("inertia", "K-Means Elbow Curve", "Inertia", "elbow_curve.png", "steelblue"),
        (
            "silhouette",
            "Silhouette Score vs Number of Clusters",
            "Silhouette score",
            "silhouette_scores.png",
            "teal",
        ),
    ]:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(metrics["clusters"], metrics[column], marker="o", color=color)
        axis.set(title=title, xlabel="Number of clusters (K)", ylabel=ylabel)
        figure.tight_layout()
        figure.savefig(figures_dir / filename, dpi=160)
        plt.close(figure)

    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    customer_data["KMeans Cluster"] = kmeans.fit_predict(features)
    customer_data.to_csv(reports_dir / "customer_cluster_assignments.csv", index=False)
    cluster_sizes = customer_data["KMeans Cluster"].value_counts().sort_index()
    print(f"\nK-Means selected K={selected_k}; cluster sizes:\n{cluster_sizes.to_string()}")

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=customer_data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="KMeans Cluster",
        palette="tab10",
        s=65,
        ax=axis,
    )
    axis.set_title("Mall Customers: K-Means Segments")
    figure.tight_layout()
    figure.savefig(figures_dir / "kmeans_customer_clusters.png", dpi=160)
    plt.close(figure)

    hierarchy_scores: list[dict[str, float | int]] = []
    for cluster_count in cluster_counts:
        labels = AgglomerativeClustering(n_clusters=cluster_count).fit_predict(features)
        hierarchy_scores.append(
            {
                "clusters": cluster_count,
                "silhouette": float(silhouette_score(features, labels)),
            }
        )
    hierarchy_metrics = pd.DataFrame(hierarchy_scores)
    hierarchy_metrics.to_csv(reports_dir / "hierarchical_silhouette_scores.csv", index=False)
    hierarchical_k = int(
        hierarchy_metrics.loc[hierarchy_metrics["silhouette"].idxmax(), "clusters"]
    )
    hierarchical_labels = AgglomerativeClustering(n_clusters=hierarchical_k).fit_predict(features)
    ari = float(adjusted_rand_score(customer_data["KMeans Cluster"], hierarchical_labels))
    print(
        f"Agglomerative selected K={hierarchical_k}; "
        f"silhouette={hierarchy_metrics['silhouette'].max():.3f}; "
        f"adjusted Rand index vs K-Means={ari:.3f}"
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    dendrogram(linkage(features, method="ward"), truncate_mode="lastp", p=30, ax=axis)
    axis.set(title="Hierarchical Clustering Dendrogram (last 30 merges)", ylabel="Ward distance")
    figure.tight_layout()
    figure.savefig(figures_dir / "hierarchical_dendrogram.png", dpi=160)
    plt.close(figure)

    pca = PCA(random_state=42).fit(features)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    axis.axhline(0.95, color="gray", linestyle="--", label="95% variance")
    axis.set(
        title="PCA Cumulative Explained Variance",
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance ratio",
        ylim=(0, 1.05),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "pca_explained_variance.png", dpi=160)
    plt.close(figure)

    pca_2d = PCA(n_components=2, random_state=42)
    components = pca_2d.fit_transform(features)
    pca_frame = pd.DataFrame(components, columns=["Principal Component 1", "Principal Component 2"])
    pca_frame["KMeans Cluster"] = customer_data["KMeans Cluster"].to_numpy()
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_frame,
        x="Principal Component 1",
        y="Principal Component 2",
        hue="KMeans Cluster",
        palette="tab10",
        s=55,
        ax=axis,
    )
    axis.set_title("Customer Segments Projected onto Two Principal Components")
    figure.tight_layout()
    figure.savefig(figures_dir / "pca_2d_clusters.png", dpi=160)
    plt.close(figure)

    return {
        "selected_k": selected_k,
        "kmeans_silhouette": float(metrics.loc[metrics["clusters"] == selected_k, "silhouette"].iloc[0]),
        "cluster_sizes": cluster_sizes.to_dict(),
        "hierarchical_k": hierarchical_k,
        "hierarchical_silhouette": float(
            hierarchy_metrics.loc[hierarchy_metrics["clusters"] == hierarchical_k, "silhouette"].iloc[0]
        ),
        "adjusted_rand_index": ari,
        "explained_variance_ratio": pca_2d.explained_variance_ratio_.tolist(),
        "two_component_variance": float(pca_2d.explained_variance_ratio_.sum()),
        "records": len(customer_data),
        "cluster_table": metrics.to_dict(orient="records"),
        "hierarchical_cluster_table": hierarchy_metrics.to_dict(orient="records"),
    }