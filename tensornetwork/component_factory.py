from tensornetwork.network_components import CopyNode, Edge, Node

_COMPONENTS = {
    "Node": Node,
    "CopyNode": CopyNode,
    "Edge": Edge,
}


def get_component(name):
    if name not in _COMPONENTS:
        raise ValueError("Component {} does not exist".format(name))
    return _COMPONENTS[name]
