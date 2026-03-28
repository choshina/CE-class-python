from __future__ import annotations
import time
from collections import deque
from typing import Optional

import torch

from ceclass.formula.stl_node import STLNode
from ceclass.strategies.base import BaseClassifier, ClassificationResult


class BFSClassifier(BaseClassifier):
    """
    BFS queue-based classification strategy (baseline).

    Bottom-up walk: queue starts at **minima** (nodes with no weaker
    ``smaller_imme`` children, e.g. leaves), then follows ``greater_imme``
    toward stronger formulas when a node is covered; on failure, deactivates
    ``greater_all`` (stronger ancestors).

    This differs from MATLAB ``MyClassProblem.m``, which seeds the queue from
    ``graph.maxima`` and walks downward.
    """

    def solve(self) -> ClassificationResult:
        t_start = time.time()

        minima = [n for n in self.graph.nodes if len(n.smaller_imme) == 0]
        queue = deque(minima)
        seen_ids = {n.formula.id for n in minima}

        while queue:
            cur = queue.popleft()

            if not cur.active:
                continue

            satisfied, result = self._test_node(cur)

            if satisfied:
                cur.add_to_results(result)
                for nd in cur.greater_imme:
                    if nd.active and nd.formula.id not in seen_ids:
                        queue.append(nd)
                        seen_ids.add(nd.formula.id)
            else:
                for nd in cur.greater_all:
                    nd.active = False

        time_class = time.time() - t_start
        return self._build_result(time_class)
