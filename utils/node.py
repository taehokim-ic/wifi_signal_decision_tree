from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    attribute: Optional[int]
    value: Optional[int]
    left: Optional["Node"]
    right: Optional["Node"]
    is_leaf: bool
