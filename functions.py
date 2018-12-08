from typing import List, NewType, Union, Dict

import numpy
from functools import singledispatch
import itertools


class Hard: pass


class SymmetricHard: pass


class Saturating: pass


class SymmetricSaturating: pass


class HyperbolicTangent: pass


class Positive: pass


NeuronOutput = NewType("NeuronOutput", Union[int, float])
NetInput = NewType("NetInput", Union[int, float])
Matrix = NewType("Matrix", numpy.matrix)


@singledispatch
def limit(n: NetInput) -> NeuronOutput:
    """
    :param n: Net input
    :return: Scalar neuron output
    """
    if n >= 0:
        return 1
    else:
        return 0


@limit.register(SymmetricHard)
def _(this, n: NetInput) -> NeuronOutput:
    """

    :param this: Limit type
    :param n: Net input
    :return: Scalar neuron output
    """
    if n >= 0:
        return 1
    else:
        return -1


@singledispatch
def linear(n: NetInput) -> NeuronOutput:
    """
    Linear transfer function
    :param n: Net input
    :return: Scalar neuron output
    """
    return n


@linear.register(Saturating)
def _(this, n: NetInput) -> NeuronOutput:
    """
    Saturating linear transfer
    function
    :param n: Net input
    :return: Scalar neuron output
    """
    if 0 <= n <= 1:
        return n
    elif n < 0:
        return 0
    else:
        return 1


@linear.register(SymmetricSaturating)
def _(this, n: NetInput) -> NeuronOutput:
    """
    :param this: Linear type
    :param n: Net input
    :return: Scalar neuron output
    """
    if 1 >= n >= -1:
        return n
    elif n < -1:
        return -1
    else:
        return 1


@linear.register(Positive)
def _(this, n: NetInput) -> NeuronOutput:
    """
    :param this: Linear type
    :param n: Net input
    :return: Scala neuron output
    """
    if 0 <= n:
        return n
    else:
        return 0


@singledispatch
def sigmoid(n: NetInput) -> NeuronOutput:
    """
    :param n: Net input
    :return: Scalar neuron output
    """
    e = numpy.euler_gamma
    return 1 / (1 + e) ** -n


@sigmoid.register(HyperbolicTangent)
def _(this, n: NetInput) -> NeuronOutput:
    """
    :param this: Sigmoid type
    :param n: Net input
    :return: Scalar neuron output
    """
    e = numpy.euler_gamma
    return (e ** n - e ** -n) / (e ** n + e ** -n)


def competitive(n: Matrix) -> Matrix:
    return n
