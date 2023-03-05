from numeric_grad import Numeric


class TestNumeric:
    x = Numeric(5)
    y = Numeric(-3)
    n = 10

    def test_summation(self):
        assert (self.x + self.y).value == 2
        assert (self.x + self.n).value == 15
        assert (self.n + self.x).value == 15

    def test_multiplicattion(self):
        assert (self.x * self.y).value == -15
        assert (self.x * self.n).value == 50
        assert (self.n * self.x).value == 50
