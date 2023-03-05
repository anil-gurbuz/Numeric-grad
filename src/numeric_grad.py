from __future__ import annotations


class Numeric:
    """
    Base class for tracking numeric values
    """

    def __init__(
        self,
        value: float,
        parents: tuple[Numeric, Numeric] | None = None,
        # grad: float | None = None,
    ):
        """

        :param value: Numeric value
        :type value: float
        """
        self.value = value
        self.parents = parents
        # self.grad = grad

    def __repr__(self):
        return f"Numeric(value={self.value})"

    def __add__(self, other: float | Numeric) -> Numeric:
        """self + other

        :param other: Value to sum
        :type other: float | Numeric
        :return: Summed up Numeric type
        :rtype: Numeric
        """
        if type(other) != Numeric:
            out = Numeric(value=other + self.value)
            out.parents = (self, Numeric(other))
        else:
            out = Numeric(value=self.value + other.value)
            out.parents = (self, other)

        return out

    def __radd__(self, other: float | Numeric):
        """other + self

        :param other: Value to sum
        :type other: float | Numeric
        :return: Summed up Numeric type
        :rtype: Numeric
        """
        return self + other

    def __mul__(self, other: float | Numeric):
        """self * other

        :param other: value to multiply
        :type other: float | Numeric
        :return: Multiplied Numeric
        :rtype: Numeric
        """

        if type(other) != Numeric:
            out = Numeric(value=other * self.value)
            out.parents = (self, Numeric(other))
        else:
            out = Numeric(value=self.value * other.value)
            out.parents = (self, other)

        return out

    def __rmul__(self, other):
        """other * self

        :param other: value to multiply
        :type other: float | Numeric
        :return: Multiplied Numeric
        :rtype: Numeric
        """
        return self * other
