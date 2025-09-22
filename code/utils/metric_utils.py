import pandas as pd
from sklearn.metrics import confusion_matrix


def get_regex_performance(predictions_df, labels_df):
    """
    Compute precision, recall, and F1 scores for data citation predictions.

    Args:
        predictions_df: DataFrame with columns ['article_id', 'dataset_id', ...]
        labels_df: DataFrame with columns ['article_id', 'dataset_id', ...]

    Returns:
        dict: Dictionary containing precision, recall, and f1 scores
    """
    pred_set = set(zip(predictions_df["article_id"], predictions_df["dataset_id"]))
    true_set = set(zip(labels_df["article_id"], labels_df["dataset_id"]))

    tp = len(pred_set & true_set)  # True positives: predicted and actually present
    fp = len(pred_set - true_set)  # False positives: predicted but not actually present
    fn = len(true_set - pred_set)  # False negatives: actually present but not predicted

    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def get_metrics(predictions_df, labels_df):
    """
    Compute precision, recall, and F1 scores for data citation predictions.

    Args:
        predictions_df: DataFrame with columns ['row_id', 'article_id', 'dataset_id', 'type']
        labels_df: DataFrame with columns ['article_id', 'dataset_id', 'type']

    Returns:
        dict: Dictionary containing precision, recall, and f1 scores
    """
    hits_df = labels_df.merge(predictions_df, on=["article_id", "dataset_id", "type"])
    tp = hits_df.shape[0]
    fp = predictions_df.shape[0] - tp
    fn = labels_df.shape[0] - tp

    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def get_confusion_matrix(predictions_df, labels_df):
    """
    Compute confusion matrix for data citation predictions.
    """
    all_combinations = set()
    all_combinations.update(zip(labels_df["article_id"], labels_df["dataset_id"]))
    all_combinations.update(
        zip(predictions_df["article_id"], predictions_df["dataset_id"])
    )

    comparison_data = []
    for article_id, dataset_id in all_combinations:
        true_label = labels_df[
            (labels_df["article_id"] == article_id)
            & (labels_df["dataset_id"] == dataset_id)
        ]["type"].values
        true_type = true_label[0] if len(true_label) > 0 else "NA"

        pred_label = predictions_df[
            (predictions_df["article_id"] == article_id)
            & (predictions_df["dataset_id"] == dataset_id)
        ]["type"].values
        pred_type = pred_label[0] if len(pred_label) > 0 else "NA"

        comparison_data.append(
            {
                "article_id": article_id,
                "dataset_id": dataset_id,
                "true_type": true_type,
                "pred_type": pred_type,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    labels = ["Primary", "Secondary", "NA"]
    y_true = comparison_df["true_type"]
    y_pred = comparison_df["pred_type"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\nConfusion Matrix:")
    print("Predicted ->")
    print("True ↓")
    print(f"{'':>12} {'Primary':>8} {'Secondary':>9} {'NA':>8}")
    for i, true_label in enumerate(labels):
        row = f"{true_label:>12}"
        for j, pred_label in enumerate(labels):
            row += f"{cm[i, j]:>8}"
        print(row)
