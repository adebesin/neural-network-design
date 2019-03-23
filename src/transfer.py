from typing import NewType, Union
import numpy
import itertools
from functools import singledispatch
import math

NeuronOutput = NewType("NeuronOutput", Union[int, float])


class HardLim:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        if value >= 0:
            self._output = 1
        else:
            self._output = 0

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class HardLims:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        self._output = value
        if value >= 0:
            self._output = 1
        else:
            self._output = -1

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class PureLin:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        self._output = value

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class SatLin:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        if 0 <= value <= 1:
            self._output = value
        elif value < 0:
            self._output = 0
        else:
            self._output = 1

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class SatLins:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        if 1 >= value >= -1:
            self._output = value
        elif value < -1:
            self._output = -1
        else:
            self._output = 1

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class LogSig:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        e = numpy.e
        dividend: int = 1
        divisor: float = numpy.power((1 + e), -value)
        quotient: float = dividend / divisor
        self._output = quotient

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class TanSig:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        e: float = numpy.e
        dividend: float = (
                numpy.power(e, value) -
                numpy.power(e, -value)
        )
        divisor: float = (
                numpy.power(e, value) +
                numpy.power(e, -value)
        )
        quotient: float = dividend / divisor
        self._output = quotient

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output


class PosLin:
    def __init__(
            self,
            output
    ):
        self.output = output

    @property
    def output(self) -> NeuronOutput:
        return self._output

    @output.setter
    def output(self, value) -> None:
        if 0 <= value:
            self._output = value
        else:
            self._output = 0

    @output.getter
    def output(self) -> NeuronOutput:
        return self._output

    @output.deleter
    def output(self) -> None:
        del self._output

    def v(self):
        return self.output

# TODO make all return values matrices

# E2.3

bias: float = 0
weight = numpy.matrix([3, 2])
p = numpy.array([-5, 7])
pT = p.reshape([2, 1])
netinput: numpy.matrix = weight * pT + bias

hardlims = HardLims(netinput)
hardlim = HardLim(netinput)
purelin = PureLin(netinput)
satlin = SatLin(netinput)
satlins = SatLins(netinput)
tansig = TanSig(netinput)
logsig = LogSig(netinput)
poslin = PosLin(netinput)

hardlims.output
hardlim.output
satlin.output
satlins.output
tansig.output
purelin.output
logsig.output
poslin.output

type(weight)
