from decision_tree.wifi_signal_decision_tree import WifiSignalDecisionTree
from decision_tree.evaluate import kFold_decision_tree_evaluation
from utils.data_transformer import DataTransformer
from utils.visualise_metrics import print_metrics

decision_tree = WifiSignalDecisionTree()

# Build decision tree.
transformed_data = DataTransformer.transform_dataset_to_ndarray(
    "./wifi_db/noisy_dataset.txt"
)

root, _ = decision_tree._decision_tree_learning(transformed_data, 0)

# Visualize the decision tree.
decision_tree._visualize_decision_tree()
(
    confusion_matrix,
    accuracy,
    precision,
    recall,
    f1_score,
) = kFold_decision_tree_evaluation(transformed_data)
print_metrics(confusion_matrix, accuracy, precision, recall, f1_score)
