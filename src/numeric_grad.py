from __future__ import annotations


class Numeric:
    """
    Base class for tracking numeric values
    """

    def __init__(self, value: float):
        """

        :param value: Numeric value
        :type value: float
        """
        self.value = value

    def __add__(self, other: float | Numeric):
        """Summing up

        :param other: Value to sum
        :type other: float | Numeric
        :return: Summed up Numeric type
        :rtype: Numeric
        """

        out = Numeric(value=other + self.value)
        return out
