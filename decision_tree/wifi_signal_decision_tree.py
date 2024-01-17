from .decision_tree import DecisionTree
from utils.types import *
from utils.decision_tree_visualizer import DecisionTreeVisualizer


class WifiSignalDecisionTree(DecisionTree):
    """
    A decision tree classifier on room number based on WiFi signal data
    """

    def __init__(self, label_index=7) -> None:
        """
        Initialise with an index for referencing the label column in the dataset.
        Additionally, setup a tree whose root can be referenced once decision tree has been constructed.
        """
        self.__label_index = label_index
        self.__tree = None

    def _visualize_decision_tree(self) -> None:
        """
        Calls utility function to help visualise the decision tree.
        """
        if self.__tree:
            DecisionTreeVisualizer._visualize_decision_tree(self.__tree)

    @classmethod
    def _predict(cls, test_data: ndarray, node: Node) -> List[float64]:
        """
        Given a list of datapoints, returns a list of predictions on each
        datapoint.

        Traverses the decision tree and predicts the value from the input.
        Keeps traversing the tree until it reaches a leaf node and returns the value.

        PRE: The decision tree must have already been constructed prior to calling.

        data:    value to make prediction on using decision tree.

        test_data:    list of datapoints to make predictions on.
        """
        y_pred = []
        for data in test_data:
            cur = node
            while not cur.is_leaf:
                if data[cur.attribute] < cur.value:
                    cur = cur.left
                else:
                    cur = cur.right
            y_pred.append(cur.value)

        return y_pred

    def _decision_tree_learning(
        self, training_dataset: ndarray, depth: int
    ) -> Tuple[Node, int]:
        """
        Recursively builds a decision tree based on
        the provided training dataset.

        training_dataset:   The training dataset used to build the
                            decision tree.
        depth:              Tracks the current depth of the tree.
        """

        # if all samples don't have the same label
        if not self.__is_same_sample(training_dataset):
            attribute, value, left_data, right_data = self.__find_split(
                training_dataset
            )
            t_node = Node(
                attribute=attribute, value=value, left=None, right=None, is_leaf=False
            )

            if not self.__tree:
                self.__tree = t_node

            lt_node, lt_depth = self._decision_tree_learning(left_data, depth + 1)
            rt_node, rt_depth = self._decision_tree_learning(right_data, depth + 1)

            t_node.left, t_node.right = lt_node, rt_node
            return t_node, max(lt_depth, rt_depth)

        # if all samples have the same label
        else:
            label = training_dataset[0][self.__label_index]

            return (
                Node(attribute=None, value=label, left=None, right=None, is_leaf=True),
                depth,
            )

    def __find_split(self, dataset: ndarray) -> Tuple[int, float, ndarray, ndarray]:
        """
        Selects the attribute and value that yield the highest
        information gain when splitting the dataset and returns
        them along with the resulting left and right branches.

        dataset:    The input data that needs to be split into
                    two branches.
        """

        attribute, value, left_data, right_data = 0, 0, array([]), array([])
        max_info_gain = 0

        for column in range(self.__label_index):
            sorted_data = dataset[dataset[:, column].argsort()]
            values = unique(sorted_data[:, column])

            for val in values:
                left_split, right_split = (
                    sorted_data[sorted_data[:, column] < val],
                    sorted_data[sorted_data[:, column] >= val],
                )

                info_gain = self.__calculate_info_gain(
                    sorted_data, left_split, right_split
                )

                if info_gain > max_info_gain:
                    attribute, value = column, val
                    left_data, right_data = left_split, right_split
                    max_info_gain = info_gain

        return attribute, value, left_data, right_data

    def __calculate_h(self, dataset: ndarray) -> float:
        """
        Calculates the entropy (h) of the dataset.

        dataset:    The input data to calculate the entropy of.
        """

        size, h = len(dataset), 0
        freq_counter = self.__count_label_freq(dataset)

        for _, freq in freq_counter.items():
            p = freq / size
            h += -p * log2(p)

        return h

    def __calculate_info_gain(
        self, dataset: ndarray, left: ndarray, right: ndarray
    ) -> float:
        """
        Calculates the information gain based on the entropy of
        the original dataset and the remainder after the split.

        left:   Subset of the dataset containing samples with attribute
                values less than or equal to the chosen threshold.
        right:  Subset of the dataset containing samples with attribute
                values greater than the chosen threshold.
        """

        h = self.__calculate_h(dataset)
        remainder = self.__calculate_remainder(left, right)

        return h - remainder

    def __calculate_remainder(self, left: ndarray, right: ndarray) -> float:
        """
        Calculates the remainder of a split, which is
        used in information gain calculations.

        left:   The resulting subset on the left branch.
        right:  The resulting subset on the right branch.
        """

        number_of_left_samples, number_of_right_samples = len(left), len(right)
        total_child_samples = number_of_left_samples + number_of_right_samples

        left_proportion = number_of_left_samples / total_child_samples
        right_proportion = number_of_right_samples / total_child_samples

        return left_proportion * self.__calculate_h(
            left
        ) + right_proportion * self.__calculate_h(right)

    def __is_same_sample(self, dataset: ndarray) -> bool:
        """
        Checks whether all samples in given dataset
        have same label.

        dataset:    The input data to check.
        """

        for i in range(len(dataset)):
            if dataset[i][self.__label_index] != dataset[i - 1][self.__label_index]:
                return False

        return True

    def __count_label_freq(self, dataset: ndarray) -> Dict[int, int]:
        """
        Counts the frequency of each label in the dataset
        and returns a dictionary where the keys are labels
        and the values are the corresponding counts.

        dataset:    The input data to count the labels.
        """

        freq_counter = {}
        data = dataset[:, self.__label_index]

        for label in data:
            freq_counter[label] = freq_counter.get(label, 0) + 1

        return freq_counter
