import math
import attr
import random
from typing import TypeAlias, Callable

IntegralFunc: TypeAlias = Callable[[float, float], float]

@attr.frozen
class FirstGenerator:
    lam: float

    def generate_number(self) -> float:
        return (-1 / self.lam) * math.log(random.random())

    def calculate_distribution(self, x: float) -> float:
        return 1 - math.exp(-self.lam * x)

@attr.frozen
class SecondGenerator:
    alpha: float
    sigma: float

    def generate_number(self) -> float:
        u = sum([random.random() for _ in range(1, 13)]) - 6
        return self.sigma * u + self.alpha

    def calculate_distribution(self, x: float) -> float:
        return ( 1 + math.erf((x - self.alpha)/(math.sqrt(2) * self.sigma))) / 2

@attr.mutable
class ThirdGenerator:
    a: float
    c: float
    z: float = attr.field(factory=random.random)

    def generate_number(self) -> float:
        self.z = math.fmod(self.a * self.z, self.c)
        return self.z / self.c

    def calculate_distribution(self, x: float) -> float:
        if x < 0:
            return 0
        elif x > 1:
            return 1
        else:
            return x
