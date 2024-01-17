def print_confusion_matrix(matrix):
    for row in matrix:
        print(row)


def print_metrics(confusion_matrix, accuracy, precision, recall, f1_score):
    print("Confusion matrix: ")
    print_confusion_matrix(confusion_matrix)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {dict(precision)}")
    print(f"Recall: {dict(recall)}")
    print(f"F1 score: {dict(f1_score)}")
