import graphviz
import math
import typing


class Value:

    def __init__(self, data: typing.Union[float, int], children=None, op=None,
                 label="", backward=lambda: None):
        self.children = children if children is not None else ()
        self.op = op
        self.label = label
        self.grad = 0.0
        # _backward propagates gradients from parents to children.
        self._backward = lambda: None

        if isinstance(data, float):
            self.data = data
            return
        if isinstance(data, int):
            self.data = float(data)
            return

        raise TypeError("Values can only be floats or ints.")

    def __repr__(self):
        return f"Value(data={self.data})"

    def backward(self):
        nodes = []
        visited = set()
        def dfs(v):
            nodes.append(v)
            visited.add(v)
            for child in v.children:
                if child not in visited:
                    dfs(child)
        dfs(self)

        self.grad = 1
        for v in nodes:
            v._backward()

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return add(self, mul(Value(-1), other))

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        # truediv is normal division ie 2/3 = 0.666... not 0.
        return mul(self, pow(other, Value(-1)))

    def __pow__(self, other):
        return power(self, other)

    def __neg__(self):
        return mul(Value(-1), self)

    def tanh(self):
        # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, children=(self,), op="tanh")

        def _backward():
            # d/dx[ tanh(x) ] = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out


def add(a: Value, b: Value):
    out = Value(a.data + b.data, children = (a, b), op="+")

    def _backward():
        # _backward propagates the gradients to the children. Because of chain
        # rule. _backward() is always of the form:
        # def _backward():
        #   child_1.grad += <local gradient w/ respect to child_1> * out.grad
        #   child_2.grad += <local gradient w/ respect to child_2> * out.grad
        a.grad += out.grad
        b.grad += out.grad
    out._backward = _backward

    return out


def mul(a: Value, b: Value):
    out = Value(a.data * b.data, children=(a, b), op="*")

    def _backward():
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    out._backward = _backward

    return out


def power(a: Value, b:Value):
    out = Value(a.data ** b.data, children=(a, b), op="%.2f^%.2f" % (a.data, b.data))

    def _backward():
        a.grad += b.data * (a.data ** (b.data - 1)) * out.grad
        b.grad += (a.data ** b.data) * math.log(a.data) * out.grad
    out._backward = _backward

    return out


def trace(root: Value):
    """Return graph of children from root.

    Find all the children, and all the edges connecting these values.

    Returns pair of sets. First set are all the values. The second set is a set
    of pairs of values which are the edges between the values.
    """
    seen, edges = set(), set()
    todo = [root]
    seen.add(root)
    while todo:
        curr = todo.pop()
        for child in curr.children:
            # Since duplicate nodes are never popped from todo
            # adding all the edges from curr never adds duplicate edges.
            # Since in neural nets values from from the child to the parents
            # add an edge from child to parent.
            edges.add((child, curr))
            if child not in seen:
                seen.add(child)
                # Duplicate notes are never added to todo because they are
                # always checked if they have been seen before.
                todo.append(child)
    return seen, edges


def draw_dot(root: Value):
    graph = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for node in nodes:
        # For each Value create a node.
        uid = str(id(node))
        graph.node(uid, label="{%s | data: %.4f | grad: %.4f}" % (
            node.label, node.data, node.grad), shape="record")
        if node.op:
            # If the Value is also an op create another node for op, and then
            # connect from op to value node.
            graph.node(uid+node.op, label=node.op)
            graph.edge(uid+node.op, uid)
    for head, tail in edges:
        head_uid = str(id(head))
        tail_uid = str(id(tail)) + tail.op
        graph.edge(head_uid, tail_uid)
    return graph
