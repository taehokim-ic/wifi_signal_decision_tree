from abc import ABC, abstractclassmethod, abstractmethod
from utils.types import *


class DecisionTree(ABC):
    @abstractmethod
    def _decision_tree_learning(
        self, training_dataset: ndarray, depth: int
    ) -> Tuple[Node, int]:
        pass

    @abstractmethod
    def _predict(self, test_data: ndarray) -> List[float64]:
        pass

    @abstractclassmethod
    def _visualize_decision_tree(cls, root: Node) -> None:
        pass
