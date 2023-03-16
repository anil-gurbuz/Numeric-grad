import math

import torch
import torch.nn.functional as F

from numeric_grad.numeric_grad import Numeric


class TestNumeric:
    x = Numeric(5)
    y = Numeric(-3)
    n = 10
    z = x + y
    out = z * n

    a = Numeric(7)
    b = Numeric(3)
    c = -4
    d = a - b  # d = 4
    e = d ** (1 / 2)  # e = 2
    f = e / c  #  f = -0.5

    def test_summation(self):
        assert self.z.value == 2
        assert (self.x + self.n).value == 15
        assert (self.n + self.x).value == 15

        ta = torch.tensor(5.0, requires_grad=True)
        tb = torch.tensor(-15.0, requires_grad=True)
        tc = ta + tb
        tc.backward()

        na = Numeric(5.0)
        nb = Numeric(-15.0)
        nc = na + nb
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_multiplication(self):
        assert self.out.value == 20
        assert (self.x * self.n).value == 50
        assert (self.n * self.x).value == 50

        ta = torch.tensor(5.0, requires_grad=True)
        tb = torch.tensor(-15.0, requires_grad=True)
        tc = ta * tb
        tc.backward()

        na = Numeric(5.0)
        nb = Numeric(-15.0)
        nc = na * nb
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_subtraction(self):
        assert self.d.value == 4
        assert (self.c - self.b).value == -7

        ta = torch.tensor(5.0, requires_grad=True)
        tb = torch.tensor(-15.0, requires_grad=True)
        tc = ta - tb
        tc.backward()

        na = Numeric(5.0)
        nb = Numeric(-15.0)
        nc = na - nb
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_division(self):
        assert self.f.value == -0.5
        assert (self.a / self.c).value == -7 / 4
        assert (self.c / self.a).value == -4 / 7

        ta = torch.tensor(5.0, requires_grad=True)
        tb = torch.tensor(-15.0, requires_grad=True)
        tc = ta / tb
        tc.backward()

        na = Numeric(5.0)
        nb = Numeric(-15.0)
        nc = na / nb
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_power(self):
        assert self.e.value == 2
        assert (self.c**self.b).value == -64
        ta = torch.tensor(5.0, requires_grad=True)
        tb = torch.tensor(-15.0, requires_grad=True)
        tc = ta**tb
        tc.backward()

        na = Numeric(5.0)
        nb = Numeric(-15.0)
        nc = na**nb
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_sigmoid(self):
        assert math.isclose(
            self.f.sigmoid().value,
            F.sigmoid(torch.tensor(self.f.value)).item(),
            rel_tol=0.0001,
        )
        assert math.isclose(
            self.b.sigmoid().value,
            F.sigmoid(torch.tensor(self.b.value)).item(),
            rel_tol=0.0001,
        )

        ta = torch.tensor(5.0, requires_grad=True)
        tc = torch.sigmoid(ta)
        tc.backward()

        na = Numeric(5.0)
        nc = na.sigmoid()
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_tanH(self):
        assert math.isclose(
            self.f.tanH().value,
            F.tanh(torch.tensor(self.f.value)).item(),
            rel_tol=0.0001,
        )
        assert math.isclose(
            self.b.tanH().value,
            F.tanh(torch.tensor(self.b.value)).item(),
            rel_tol=0.0001,
        )

        ta = torch.tensor(5.0, requires_grad=True)
        tc = torch.tanh(ta)
        tc.backward()

        na = Numeric(5.0)
        nc = na.tanH()
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.001)

    def test_ReLU(self):
        assert math.isclose(
            self.f.ReLU().value,
            F.relu(torch.tensor(self.f.value)).item(),
            rel_tol=0.0001,
        )
        assert math.isclose(
            self.b.ReLU().value,
            F.relu(torch.tensor(self.b.value)).item(),
            rel_tol=0.0001,
        )

        ta = torch.tensor(5.0, requires_grad=True)
        tc = torch.relu(ta)
        tc.backward()

        na = Numeric(5.0)
        nc = na.ReLU()
        nc.grad = 1
        nc._pass_local_derivative()
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_leaky_ReLU(self):
        assert math.isclose(
            self.f.leaky_ReLU(leakage_slope=0.01).value,
            F.leaky_relu_(
                torch.tensor(float(self.f.value)), negative_slope=0.01
            ).item(),
            rel_tol=0.0001,
        )
        assert math.isclose(
            self.b.leaky_ReLU(leakage_slope=0.01).value,
            F.leaky_relu_(
                torch.tensor(float(self.b.value)), negative_slope=0.01
            ).item(),
            rel_tol=0.0001,
        )

        ta = torch.tensor(5.0, requires_grad=True)
        tc = torch.nn.LeakyReLU(negative_slope=0.01)(ta)
        tc.backward()

        na = Numeric(5.0)
        nc = na.leaky_ReLU(leakage_slope=0.01)
        nc.grad = 1
        nc._pass_local_derivative()
        print(ta.grad.item(), na.grad)
        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)

    def test_parse_till_leaf(self):
        a = Numeric(3.90)
        b = Numeric(4.0)
        c = a + b
        d = c * c
        root_to_leaf = list(reversed(Numeric.parse_till_leaf(d)))
        assert len(root_to_leaf) == 4
        assert root_to_leaf[0] == d
        assert root_to_leaf[-1] in [a, b]

    def test_zero_grad(self):
        a = Numeric(3.90)
        b = Numeric(4.0)
        c = a + b
        d = c * c
        # d.zero_grad()
        d.grad = 1
        d._pass_local_derivative()
        d.zero_grad()

        assert d.grad == 0
        assert a.grad == 0
        assert c.grad == 0

    def test_global_gradient_flow(self):
        ta = torch.tensor(5.0, requires_grad=True)
        tb = torch.tensor(-15.0, requires_grad=True)
        tc = ta + tb
        tc.retain_grad()
        td = tc * tc
        td.backward()

        na = Numeric(5.0)
        nb = Numeric(-15.0)
        nc = na + nb
        nd = nc * nc
        nd.zero_grad()
        nd.backward()

        assert math.isclose(ta.grad.item(), na.grad, rel_tol=0.0001)
        assert math.isclose(tb.grad.item(), nb.grad, rel_tol=0.0001)
        assert math.isclose(tc.grad.item(), nc.grad, rel_tol=0.0001)
