# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A TensorNetwork for counting 3SAT solutions.

This is an implementation of https://arxiv.org/abs/1105.3201.

3SAT problems are boolean satisfiability problem of the form
(a OR NOT b OR c) AND (NOT a OR NOT c OR d) AND ...
where a, b, c, d are variables that can take the values True or False.
See https://en.wikipedia.org/wiki/Boolean_satisfiability_problem for a more
in-depth description.

3SAT TensorNetworks are networks that can find/count all solutions to a
given 3SAT problem. At a high level, these networks are constructed by
connecting "clause nodes" to "variable nodes" through "copy tensors".

Clause nodes are tensors of shape (2, 2, 2) with 1 for every variable
assigment that satifies the clause, and 0 for the one assigment that doesn't.
For example, for the clause (a OR b OR NOT c), then
clause.get_tensor()[0][0][1] == 0, and is 1 everywhere else.

Variable Nodes are (2, 1) tensors that have [1, 1] as their concrete value.
You can think if this node like an unnormalized superposition of the variable
being True and False.

Copy tensors are tensors of shape (2, 2, 2). These tensors have value 1 at
positions [1][1][1] and [0][0][0] and have value 0 eveywhere else.
"""

import numpy as np
from typing import List, Tuple, Set
import tensornetwork as tn


def sat_tn(clauses: List[Tuple[int, int, int]]) -> List[tn.Edge]:
    """Create a 3SAT TensorNetwork of the given 3SAT clauses.

      After full contraction, this network will be a tensor of size (2, 2, ..., 2)
      with the rank being the same as the number of variables. Each element of the
      final tensor represents whether the given assignment satisfies all of the
      clauses. For example, if final_node.get_tensor()[0][1][1] == 1, then the
      assiment (False, True, True) satisfies all clauses.

    Args:
      clauses: A list of 3 int tuples. Each element in the tuple corresponds to a
        variable in the clause. If that int is negative, that variable is negated
        in the clause.

    Returns:
      net: The 3SAT TensorNetwork.
      var_edges: The edges for the given variables.

    Raises:
      ValueError: If any of the clauses have a 0 in them.
    """
    for clause in clauses:
        if 0 in clause:
            raise ValueError("0's are not allowed in the clauses.")
    var_set = set()
    for clause in clauses:
        var_set |= {abs(x) for x in clause}
    num_vars = max(var_set)
    var_nodes = []
    var_edges = []

    # Prepare the variable nodes.
    for _ in range(num_vars):
        new_node = tn.Node(np.ones(2, dtype=np.int32))
        var_nodes.append(new_node)
        var_edges.append(new_node[0])

    # Create the nodes for each clause
    for clause in clauses:
        (
            a,
            b,
            c,
        ) = clause
        clause_tensor = np.ones((2, 2, 2), dtype=np.int32)
        clause_tensor[
            (-np.sign(a) + 1) // 2, (-np.sign(b) + 1) // 2, (-np.sign(c) + 1) // 2
        ] = 0
        clause_node = tn.Node(clause_tensor)

        # Connect the variable to the clause through a copy tensor.
        for i, var in enumerate(clause):
            copy_tensor_node = tn.CopyNode(3, 2)
            clause_node[i] ^ copy_tensor_node[0]
            var_edges[abs(var) - 1] ^ copy_tensor_node[1]
            var_edges[abs(var) - 1] = copy_tensor_node[2]

    return var_edges


def sat_count_tn(clauses: List[Tuple[int, int, int]]) -> Set[tn.AbstractNode]:
    """Create a 3SAT Count TensorNetwork.

    After full contraction, the final node will be the count of all possible
    solutions to the given 3SAT problem.

    Args:
      clauses: A list of 3 int tuples. Each element in the tuple corresponds to a
        variable in the clause. If that int is negative, that variable is negated
        in the clause.

    Returns:
      nodes: The set of nodes
    """
    var_edges1 = sat_tn(clauses)
    var_edges2 = sat_tn(clauses)
    for edge1, edge2 in zip(var_edges1, var_edges2):
        edge1 ^ edge2
    # TODO(chaseriley): Support diconnected SAT graphs.
    return tn.reachable(var_edges1[0].node1)
