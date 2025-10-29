"""
Pairwise genre classification and feature importance analysis.

This module performs binary classification between genre pairs to understand
which acoustic features are most discriminative for each genre comparison.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

sns.set_context('poster')


def load_pair_data(data_dir, genre1, genre2, target1=0, target2=1):
    """
    Load feature data for a pair of genres.

    Args:
        data_dir: Directory containing the concatenated CSV files
        genre1: Name of first genre
        genre2: Name of second genre
        target1: Label for first genre (default: 0)
        target2: Label for second genre (default: 1)

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        groups: Group identifiers for each sample (n_samples,)
        feature_names: List of feature names
    """
    # Load dataframes
    df1 = pd.read_csv(os.path.join(data_dir, f'{genre1}_concatenated.csv'))
    df2 = pd.read_csv(os.path.join(data_dir, f'{genre2}_concatenated.csv'))

    # Extract features
    X1 = df1.iloc[:, 2:-2].values.astype(np.float32)
    y1 = np.full(X1.shape[0], target1)
    groups1 = df1['group'].values

    X2 = df2.iloc[:, 2:-2].values.astype(np.float32)
    y2 = np.full(X2.shape[0], target2)
    groups2 = df2['group'].values

    # Get feature names
    feature_names = df1.iloc[:, 2:-2].columns.tolist()

    # Combine data
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    groups = np.concatenate((groups1, groups2))

    print(f"Loaded {genre1} vs {genre2}: X={X.shape}, y={y.shape}")

    return X, y, groups, feature_names


def train_and_extract_importances(clf, X, y, groups, n_splits=100, test_size=0.2, random_state=0):
    """
    Train classifier and extract feature importances across multiple folds.

    Args:
        clf: Scikit-learn classifier pipeline with feature selection
        X: Feature matrix
        y: Target labels
        groups: Group identifiers
        n_splits: Number of cross-validation splits
        test_size: Proportion of data for testing
        random_state: Random seed

    Returns:
        predictions: Tuple of (true_labels, predicted_labels)
        accuracies: List of accuracy scores
        roc_scores: List of ROC-AUC scores
        importances: Array of feature importances (n_splits, n_features)
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    predictions = [[], []]
    accuracies = []
    roc_scores = []
    importances = []

    print(f"Running {n_splits} cross-validation splits...")

    for fold, (train_index, test_index) in enumerate(gss.split(X, y, groups=groups)):
        clf.fit(X[train_index], y[train_index])
        outputs = clf.predict(X[test_index])

        # Collect metrics
        accuracies.append(accuracy_score(y[test_index], outputs))
        roc_scores.append(roc_auc_score(y[test_index], outputs))

        predictions[0].extend(y[test_index])
        predictions[1].extend(outputs)

        # Extract feature importances
        # The feature_selection step reduces features, and clf.steps[-1] is the final classifier
        importance = clf.steps[-2][1].inverse_transform(
            clf.steps[-1][1].feature_importances_[None, :]
        )
        importances.append(importance)

        if (fold + 1) % 20 == 0:
            print(f"  Completed {fold + 1}/{n_splits} folds")

    importances = np.array(importances)[:, 0, :]

    return predictions, accuracies, roc_scores, importances


def plot_feature_importances(importances, scores, feature_names, output_path, top_n=30):
    """
    Plot weighted feature importances.

    Args:
        importances: Array of feature importances (n_splits, n_features)
        scores: ROC-AUC scores for weighting
        feature_names: List of feature names
        output_path: Path to save figure
        top_n: Number of top features to display
    """
    # Calculate weighted importances
    scores = np.array(scores)
    w_importances = scores.dot(importances) / np.sum(scores)

    # Get top features
    index = np.argsort(w_importances)[::-1]
    index = index[np.where(w_importances[index] > 0.002)]
    index = index[:top_n]

    # Plot
    plt.figure(figsize=(10, 12))
    plt.plot(w_importances[index], np.arange(len(index)), '.-', markersize=10)
    plt.yticks(np.arange(len(index)), [feature_names[i] for i in index])
    plt.ylabel('Features')
    plt.xlabel('Weighted Importance Score')
    plt.title('Top Discriminative Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {output_path}")
    plt.close()

    return w_importances, index


def main():
    """Command-line interface for pairwise analysis."""
    parser = argparse.ArgumentParser(
        description='Perform pairwise genre classification and feature importance analysis'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing the concatenated CSV feature files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=100,
        help='Number of cross-validation splits (default: 100)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in ExtraTrees classifier (default: 100)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define genre pairs
    pairs = [
        ('salsa', 'soca', 'Salsa vs Soca'),
        ('salsa', 'reggae', 'Salsa vs Reggae'),
        ('soca', 'reggae', 'Soca vs Reggae')
    ]

    # Store results
    all_results = []

    for genre1, genre2, label in pairs:
        print(f"\n{'='*60}")
        print(f"Analyzing {label}")
        print(f"{'='*60}")

        # Load data
        X, y, groups, feature_names = load_pair_data(args.data_dir, genre1, genre2)

        # Create classifier with feature selection
        clf = Pipeline([
            ('std', StandardScaler()),
            ('feature_selection', SelectFromModel(
                ExtraTreesClassifier(
                    n_estimators=args.n_estimators,
                    class_weight='balanced'
                )
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=args.n_estimators,
                class_weight='balanced'
            ))
        ])

        # Train and evaluate
        predictions, accuracies, roc_scores, importances = train_and_extract_importances(
            clf, X, y, groups,
            n_splits=args.n_splits
        )

        # Calculate metrics
        median_acc = np.median(accuracies)
        median_roc = np.median(roc_scores)

        print(f"\nResults:")
        print(f"  Median Accuracy: {median_acc:.4f}")
        print(f"  Median ROC-AUC: {median_roc:.4f}")

        # Plot feature importances
        output_path = os.path.join(
            args.output_dir,
            f'feature_importance_{genre1}_vs_{genre2}.png'
        )
        w_importances, top_indices = plot_feature_importances(
            importances, roc_scores, feature_names, output_path
        )

        # Save top features to CSV
        top_features_df = pd.DataFrame({
            'feature': [feature_names[i] for i in top_indices],
            'importance': w_importances[top_indices]
        })
        csv_path = os.path.join(
            args.output_dir,
            f'top_features_{genre1}_vs_{genre2}.csv'
        )
        top_features_df.to_csv(csv_path, index=False)

        # Store results
        all_results.append({
            'comparison': label,
            'median_accuracy': median_acc,
            'median_roc_auc': median_roc,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        })

    # Save summary results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, 'pairwise_results.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved results summary to {results_path}")
    print(f"{'='*60}")
    print(results_df)


if __name__ == '__main__':
    main()
