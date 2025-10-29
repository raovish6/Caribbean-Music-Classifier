"""
Train and evaluate 3-class music genre classifiers.

This module trains multiple classifier types on acoustic features to distinguish
between salsa, soca, and reggae music genres using group-based cross-validation.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

sns.set_context('poster')


def load_data(data_dir):
    """
    Load feature data for all three genres.

    Args:
        data_dir: Directory containing the concatenated CSV files

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        groups: Group identifiers for each sample (n_samples,)
    """
    # Load dataframes
    salsa = pd.read_csv(os.path.join(data_dir, 'salsa_concatenated.csv'))
    soca = pd.read_csv(os.path.join(data_dir, 'soca_concatenated.csv'))
    reggae = pd.read_csv(os.path.join(data_dir, 'reggae_concatenated.csv'))

    # Extract features (columns 2 to -2, excluding index and metadata)
    salsa_X = salsa.iloc[:, 2:-2].values.astype(np.float32)
    salsa_Y = np.zeros(salsa_X.shape[0])
    salsa_groups = salsa['group'].values

    soca_X = soca.iloc[:, 2:-2].values.astype(np.float32)
    soca_Y = np.ones(soca_X.shape[0])
    soca_groups = soca['group'].values

    reggae_X = reggae.iloc[:, 2:-2].values.astype(np.float32)
    reggae_Y = np.full(reggae_X.shape[0], 2)
    reggae_groups = reggae['group'].values

    # Combine all data
    X = np.vstack((salsa_X, soca_X, reggae_X))
    y = np.concatenate((salsa_Y, soca_Y, reggae_Y))
    groups = np.concatenate((salsa_groups, soca_groups, reggae_groups))

    print(f"Loaded data shape: X={X.shape}, y={y.shape}, groups={groups.shape}")

    return X, y, groups


def train_and_evaluate(clf, X, y, groups, n_splits=100, test_size=0.2, random_state=0):
    """
    Train and evaluate a classifier using group-based cross-validation.

    Args:
        clf: Scikit-learn classifier pipeline
        X: Feature matrix
        y: Target labels
        groups: Group identifiers
        n_splits: Number of cross-validation splits
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        predictions: Tuple of (true_labels, predicted_labels) across all folds
        accuracies: List of accuracy scores for each fold
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    predictions = [[], []]
    accuracies = []

    print(f"Running {n_splits} cross-validation splits...")

    for fold, (train_index, test_index) in enumerate(gss.split(X, y, groups=groups)):
        clf.fit(X[train_index], y[train_index])
        outputs = clf.predict(X[test_index])

        accuracies.append(accuracy_score(y[test_index], outputs))
        predictions[0].extend(y[test_index])
        predictions[1].extend(outputs)

        if (fold + 1) % 20 == 0:
            print(f"  Completed {fold + 1}/{n_splits} folds")

    return predictions, accuracies


def plot_confusion_matrix(predictions, output_path=None):
    """
    Generate and save confusion matrix plot.

    Args:
        predictions: Tuple of (true_labels, predicted_labels)
        output_path: Path to save figure (if None, displays instead)
    """
    cm = confusion_matrix(predictions[0], predictions[1], normalize='true')
    display_labels = ['Salsa', 'Soca', 'Reggae']

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', values_format='.2f')
    plt.title('Normalized Confusion Matrix')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Command-line interface for training classifiers."""
    parser = argparse.ArgumentParser(
        description='Train 3-class music genre classifiers'
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
        '--classifier',
        type=str,
        choices=['extratrees', 'logistic', 'svm', 'mlp', 'all'],
        default='all',
        help='Classifier type to train (default: all)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=100,
        help='Number of cross-validation splits (default: 100)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    X, y, groups = load_data(args.data_dir)

    # Define classifiers
    classifiers = {}

    if args.classifier in ['extratrees', 'all']:
        classifiers['ExtraTrees'] = Pipeline([
            ('std', StandardScaler()),
            ('et', ExtraTreesClassifier(n_estimators=100, class_weight='balanced'))
        ])

    if args.classifier in ['logistic', 'all']:
        classifiers['LogisticRegression'] = Pipeline([
            ('std', StandardScaler()),
            ('logistic', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])

    if args.classifier in ['svm', 'all']:
        classifiers['SVM'] = Pipeline([
            ('std', StandardScaler()),
            ('svm', SVC(class_weight='balanced'))
        ])

    if args.classifier in ['mlp', 'all']:
        classifiers['MLP'] = Pipeline([
            ('std', StandardScaler()),
            ('mlp', MLPClassifier(max_iter=1000))
        ])

    # Train and evaluate each classifier
    results = {}

    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")

        predictions, accuracies = train_and_evaluate(
            clf, X, y, groups,
            n_splits=args.n_splits
        )

        median_acc = np.median(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        results[name] = {
            'median_accuracy': median_acc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        }

        print(f"\nResults:")
        print(f"  Median Accuracy: {median_acc:.4f}")
        print(f"  Mean Accuracy: {mean_acc:.4f}")
        print(f"  Std Accuracy: {std_acc:.4f}")

        # Save confusion matrix
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_{name.lower()}.png')
        plot_confusion_matrix(predictions, cm_path)

    # Save summary results
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(args.output_dir, 'classification_results.csv')
    results_df.to_csv(results_path)
    print(f"\n{'='*60}")
    print(f"Saved results summary to {results_path}")
    print(f"{'='*60}")
    print(results_df)


if __name__ == '__main__':
    main()
