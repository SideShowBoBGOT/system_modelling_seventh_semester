import math
import attr
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeAlias, Callable, Protocol
from scipy import stats

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

class GeneratorProtocol(Protocol):
    def generate_number(self) -> float:
        ...
    
    def calculate_distribution(self, x: float) -> float:
        ...

@attr.frozen
class CalcCountsInIntervalResult:
    data: list[int]
    interval_size: float

def calc_counts_in_interval(
    numbers: list[float],
    interval_count: np.uint32
) -> CalcCountsInIntervalResult:
    min_val = min(numbers)
    max_val = max(numbers)
    interval_size = float((max_val - min_val) / interval_count)
    counts_in_interval = [0 for _ in numbers]
    for number in numbers:
        index = int((number - min_val) / interval_size)
        if number == max_val:
            index -= 1
        counts_in_interval[index] += 1
    return CalcCountsInIntervalResult(counts_in_interval, interval_size)

class CalculateDistributionProtocol(Protocol):
    def calculate_distribution(self, x: float) -> float:
        ...

def calc_chi_value(
    calculate_distribution: Callable[[float], float],
    numbers: list[float],
    interval_count: np.uint32
):
    counts_in_interval = calc_counts_in_interval(numbers, interval_count)
    min_val = min(numbers)

    x2 = 0
    count_in_interval = 0
    left_index = 0

    for i in range(interval_count):
        count_in_interval += counts_in_interval.data[i]
        if count_in_interval < 5 and i != interval_count - 1:
            continue
        
        left = min_val + counts_in_interval.interval_size * left_index
        right = min_val + counts_in_interval.interval_size * (i + 1)
        expected_count = len(numbers) * (calculate_distribution(right) - calculate_distribution(left))
        
        x2 += (count_in_interval - expected_count) ** 2 / expected_count
        
        left_index = i + 1
        count_in_interval = 0
    return x2

@attr.frozen
class CheckChiValueRes:
    x2: float
    x2_table: float
    is_hypothesis_ok: bool 

def check_chi_value(
    calculate_distribution: Callable[[float], float],
    numbers: list[float],
    interval_count: np.uint32,
    significance_level: float
) -> CheckChiValueRes:
    x2 = calc_chi_value(calculate_distribution, numbers, interval_count)
    degrees_of_freedom = interval_count - 1 - 2
    x2_table: float = stats.chi2.ppf(1 - significance_level, degrees_of_freedom) #type: ignore
    return CheckChiValueRes(x2, x2_table, x2 < x2_table)

def calc_average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

def calc_variance(numbers: list[float]) -> float:
    avg = calc_average(numbers)
    return sum([math.pow(x - avg, 2) for x in numbers]) / len(numbers)

def analyze_generator(
    generator: GeneratorProtocol,
    number_count: np.uint32,
    interval_count: np.uint32,
    significance_level: float
) -> None:
    numbers = [generator.generate_number() for _ in range(number_count)]
    
    avg = calc_average(numbers)
    var = calc_variance(numbers)
    chi_res = check_chi_value(generator.calculate_distribution, numbers, interval_count, significance_level)
    
    plt.figure(figsize=(10, 6)) #type: ignore
    plt.hist(numbers, bins=30, edgecolor='black') #type: ignore
    plt.title(f'Histogram of {generator.__class__.__name__} (n={number_count})') #type: ignore
    plt.xlabel('Value') #type: ignore
    plt.ylabel('Frequency') #type: ignore

    plt.plot([], [], ' ', label=f'{avg=}')
    plt.plot([], [], ' ', label=f'{var=}')
    plt.plot([], [], ' ', label=f'{chi_res.x2=}')
    plt.plot([], [], ' ', label=f'{chi_res.x2_table=}')
    plt.plot([], [], ' ', label=f'{chi_res.is_hypothesis_ok=}')
    plt.legend(loc='upper right')

    
    print(f"{avg=}")
    print(f"{var=}")
    print(f"{chi_res=}")

    plt.show() #type: ignore

def main() -> None:
    number_count = np.uint32(10000)
    interval_count = np.uint32(20)
    significance_level = 0.05
    analyze_generator(
        FirstGenerator(9), number_count,
        interval_count, significance_level
    )

if __name__ == '__main__':
    main()
