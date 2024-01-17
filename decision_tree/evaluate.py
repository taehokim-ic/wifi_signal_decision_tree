from collections import defaultdict
from decision_tree.wifi_signal_decision_tree import WifiSignalDecisionTree
from utils.types import *


def kFold_decision_tree_evaluation(
    dataset: ndarray,
) -> tuple((ndarray, float, defaultdict, defaultdict, defaultdict)):
    """
    Performs 10-fold cross-validation on the dataset. On each iteration
    it selects nine folds for training and one fold for testing.

    Calls the function 'evaluate' using the trained model and test fold
    which returns the metrics describing performance on the test dataset.

    Iteratively computes metrics across ten iterations for each of the
    ten test folds, and then computes the averaged metrics using
    'compute_average_metrics'.

    Evaluation metrics include producing a confusion matrix, calculating
    accuracy, precision and recall rates per class, and F1 score.
    """
    random.shuffle(dataset)
    num_samples = len(dataset)
    fold_size = num_samples // 10

    results = []

    for i in range(10):
        test_start_index = i * fold_size
        test_end_index = (i + 1) * fold_size
        test_data = dataset[test_start_index:test_end_index]
        training_data = concatenate(
            [dataset[:test_start_index], dataset[test_end_index:]]
        )

        decision_tree = WifiSignalDecisionTree()
        trained_tree, _ = decision_tree._decision_tree_learning(training_data, 0)
        # appends confusion matrix, accuracy, precision, recall and f1 score
        results.append(evaluate(test_data, trained_tree))

    return compute_average_metrics(results)


def evaluate(
    test_db: ndarray, trained_tree: Node
) -> tuple((ndarray, float, defaultdict, defaultdict, defaultdict)):
    """
    Construct a decision tree from training data and makes predictions
    using test data.

    Use predictions to compute and return a confusion matrix, accuracy,
    precision and recalls rates per class, and F1 score.
    """

    # make predictions using trained model on test data
    y_pred = WifiSignalDecisionTree._predict(test_db, trained_tree)
    y_test = [x[-1] for x in test_db]

    confusion_matrix = compute_confusion_matrix(y_test, y_pred)
    accuracy = compute_accuracy(confusion_matrix)
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)
    f1_score = compute_f1_score(precision, recall)

    return confusion_matrix, accuracy, precision, recall, f1_score


# computes confusion matrix using predicted and actual class values
def compute_confusion_matrix(y_test: List[float], y_pred: List[float]) -> ndarray:
    confusion_matrix = zeros((4, 4), dtype=int)

    for i in range(len(y_test)):
        confusion_matrix[int(y_test[i]) - 1, int(y_pred[i]) - 1] += 1

    return confusion_matrix


# computes accuracy using confusion matrix values
def compute_accuracy(confusion_matrix: ndarray) -> float:
    total = 0
    for row in confusion_matrix:
        for col in row:
            total += col

    correct = 0
    for i in range(4):
        correct += confusion_matrix[i][i]
    return correct / total


# computes precision using confusion matrix values
def compute_precision(confusion_matrix: ndarray) -> dict:
    transposed_confusion_matrix = transpose(confusion_matrix)
    precision = {}
    for i in range(len(transposed_confusion_matrix)):
        class_total = sum(transposed_confusion_matrix[i])
        precision[i + 1] = transposed_confusion_matrix[i][i] / class_total
    return precision


# computes recall using confusion matrix values
def compute_recall(confusion_matrix: ndarray) -> dict:
    recall = {}
    for i in range(len(confusion_matrix)):
        class_total = sum(confusion_matrix[i])
        recall[i + 1] = confusion_matrix[i][i] / class_total
    return recall


# computes f1 score using precision and recall
def compute_f1_score(precision: List[float], recall: List[float]) -> dict:
    f1_score = {}
    for i in range(len(precision)):
        f1_score[i + 1] = (
            2 * precision[i + 1] * recall[i + 1] / (precision[i + 1] + recall[i + 1])
        )
    return f1_score


# provided metrics computed for each fold, it returns the averaged metrics
def compute_average_metrics(
    folds: ndarray,
) -> tuple((ndarray, float, defaultdict, defaultdict, defaultdict)):
    """
    Provided metrics calculated previously for each fold, and then averages
    the metrics by the number of folds and returns.

    folds:  2D matrix consisting of confusion matrix, accuracy, precision,
    recall and f1 score for each fold.
    """
    num_folds = len(folds)
    avg_confusion_matrix = zeros((4, 4), dtype=int)
    avg_accuracy, avg_precision, avg_recall, avg_f1_score = (
        0,
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    # sums values across each fold for each metric
    for fold in folds:
        confusion_matrix, accuracy, precision, recall, f1_score = fold
        # sums values for each class
        avg_accuracy += accuracy
        for i in range(1, 5):
            avg_precision[i] += precision[i]
            avg_recall[i] += recall[i]
            avg_f1_score[i] += f1_score[i]
        avg_confusion_matrix += confusion_matrix

    # averages values for each metric
    avg_accuracy /= num_folds
    for i in range(1, 5):
        # averages and formats metrics by class
        avg_precision[i] = f"{round(avg_precision[i] * 100 / num_folds, 2)}%"
        avg_recall[i] = f"{round(avg_recall[i] * 100 / num_folds, 2)}%"
        avg_f1_score[i] = f"{round(avg_f1_score[i] * 100 / num_folds, 2)}%"

    return (
        avg_confusion_matrix / num_folds,
        avg_accuracy,
        avg_precision,
        avg_recall,
        avg_f1_score,
    )
