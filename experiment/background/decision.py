class Decision(object):
    def __add__(self, other):
        return int(self) + other

    def __radd__(self, other):
        return other + int(self)

    def __mul__(self, other):
        return int(self) * other

    def __rmul__(self, other):
        return other * int(self)

    def __repr__(self):
        return str(self.__class__.__name__)

    def __str__(self):
        return str(self.__class__.__name__)

    def hold(self):
        return isinstance(self, Hold)

    def sell(self):
        return isinstance(self, Sell)

    def buy(self):
        return isinstance(self, Buy)


def make_decision(n: float) -> Decision:
    if n > 0:
        return Buy()
    elif n < 0:
        return Sell()
    else:
        return Hold()


class Buy(Decision):
    def __int__(self):
        return 1


class Sell(Decision):
    def __int__(self):
        return -1


class Hold(Decision):
    def __int__(self):
        return 0


if __name__ == "__main__":
    print(Hold() + 2)
    print(Hold() + Buy())
    print(2 + Buy())
    print(3 * Buy())
    print(Buy() * 2)
