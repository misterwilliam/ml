import micrograd.autodiff

import math
import typing
import unittest

def numerical_diff(f: typing.Callable[[float], float], x: float) -> float:
    """Compute derivative numerically.

    Return numercially computed derivative:
    f(x + h) - f(x) / h
    where h is some small number.
    """
    h = 0.00000001
    return (f(x + h) - f(x)) / h

class TestValue(unittest.TestCase):

    def test_add(self):
        a = micrograd.autodiff.Value(1)
        b = micrograd.autodiff.Value(2)
        c = a + b
        self.assertEqual(c.data, 3)

        c.backward()
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_sub(self):
        a = micrograd.autodiff.Value(1)
        b = micrograd.autodiff.Value(2)
        c = a - b
        self.assertEqual(c.data, -1)

        c.backward()
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, -1)

        d = a + (-b)
        self.assertEqual(d.data, c.data)
        d.backward()
        self.assertEqual(d.grad, c.grad)

    def test_mul(self):
        a = micrograd.autodiff.Value(2)
        b = micrograd.autodiff.Value(3)
        c = a * b
        self.assertEqual(c.data, 6)

        c.backward()
        self.assertEqual(a.grad, 3)
        self.assertEqual(b.grad, 2)

    def test_true_div(self):
        a = micrograd.autodiff.Value(2)
        b = micrograd.autodiff.Value(4)
        c = a / b
        self.assertEqual(c.data, 0.5)

        c.backward()
        self.assertEqual(a.grad, 1/4)
        # Derivative derived by hand.
        self.assertEqual(b.grad, -2/16)
        # Derivative derived numerically.
        self.assertAlmostEqual(b.grad, numerical_diff(lambda x: 2 / x, 4))

    def test_pow(self):
        a = micrograd.autodiff.Value(2)
        b = micrograd.autodiff.Value(3)
        c = a ** b
        self.assertEqual(c.data, 8)

        c.backward()
        self.assertEqual(a.grad, 3 * (2 ** 2))
        # Derivative derived by hand
        self.assertEqual(b.grad, (2 ** 3) * math.log(2))
        # Derivative derived numerically.
        self.assertAlmostEqual(b.grad, numerical_diff(lambda x: 2 ** x, 3))

    def test_neg(self):
        a = micrograd.autodiff.Value(3)
        b = -a
        self.assertEqual(b.data, -3)

        b.backward()
        self.assertEqual(a.grad, -1)

        # Verify that neg() is equivalent to multiplying by -1
        c = micrograd.autodiff.Value(-1) * a
        self.assertEqual(c.data, b.data)
        c.backward()
        self.assertEqual(c.grad, b.grad)

    def test_tanh(self):
        a = micrograd.autodiff.Value(0)
        b = a.tanh()
        self.assertAlmostEqual(math.tanh(a.data), b.data, places=2)

        b.backward()
        # Derivative derived numerically
        self.assertAlmostEqual(a.grad, numerical_diff(lambda x: math.tanh(x), 0))