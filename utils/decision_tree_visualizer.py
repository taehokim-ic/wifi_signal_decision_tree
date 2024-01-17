import matplotlib.pyplot as plt

from collections import deque
from numpy import random, minimum

from .node import Node

PALETTE = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "olive",
    "lime",
]


class DecisionTreeVisualizer:
    @classmethod
    def _visualize_decision_tree(cls, root: Node, palette=PALETTE) -> None:
        """
        Visualize the decision tree using matplotlib.
        """

        def _draw_node(node: Node, depth: int, x=0.5, y=0.5) -> None:
            if node.is_leaf:
                plt.text(
                    x,
                    y,
                    f"leaf:{int(node.value)}",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round", facecolor="white", edgecolor="dodgerblue"
                    ),
                    fontsize=10 - minimum(depth, 6),
                )
                return

            plt.text(
                x,
                y,
                f"[X{node.attribute}<{node.value}]",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="dodgerblue"),
                fontsize=10 - minimum(depth, 6),
            )

        if not root:
            return

        # set x and y limit of plot
        plt.xlim(-5000, 5000), plt.ylim(-300, 300)

        count, depth, q = 0, 0, deque([(root, 0, 300)])  # (node, x, y)

        while q:
            depth += 1
            for _ in range(len(q)):
                count += 1
                node, x, y = q.popleft()
                _draw_node(node, depth, x, y)
                color = palette[random.choice(len(palette))]
                if not node.is_leaf:
                    new_y = y - 30
                    if node.left:
                        x_left_node = x - 1 / 2 ** (depth + 0.3) * 6000
                        plt.plot([x, x_left_node], [y, new_y], color=color)
                        q.append((node.left, x_left_node, new_y))
                    if node.right:
                        x_right_node = x + 1 / 2 ** (depth + 0.3) * 6000
                        plt.plot([x, x_right_node], [y, new_y], color=color)
                        q.append((node.right, x_right_node, new_y))

        # Turn off axis.
        plt.axis("off")
        # Display visualization.
        plt.show()
