from typing import List, Optional, Literal, Union
import random
from random import choices
import math
import sys

def generate_exponential(time_mean: float) -> float:
    a = 0
    while a == 0:
        a = random.random()
    a = -time_mean * math.log(a)
    return a

def generate_uniform(time_min: float, time_max: float) -> float:
    a = 0
    while a == 0:
        a = random.random()
    a = time_min + a * (time_max - time_min)
    return a

def generate_normal(time_mean: float, time_deviation: float) -> float:
    a = time_mean + time_deviation * random.gauss()
    return a

def generate_erlang(time_mean: float, k: int) -> float:
    lambda_: float = k / time_mean
    sum_random_values: float = sum(math.log(random.random()) for _ in range(k))
    return (-1 / lambda_) * sum_random_values

class Customer:
    def __init__(self, enter_time: float):
        self.enter_time: float = enter_time

class Route:
    def __init__(self, element: 'Element', p: Optional[float] = None):
        self.element: 'Element' = element
        self.p: Optional[float] = p

class Element:
    next_id: int = 0

    def __init__(self, name_of_element: str = 'anonymous', delay: float = 1.0):
        self.name: str = name_of_element
        self.tnext: float = 0.0
        self.delay_mean: float = delay
        self.distribution: str = 'exp'
        self.tcurr: float = self.tnext
        self.state: int = 0
        self.id_: int = Element.next_id
        Element.next_id += 1
        self.name = 'element' + str(self.id_)
        self.delay_dev: float = 0.0
        self.quantity: int = 0
        self.route_choice: Optional[Literal['probability', 'priority']] = None
        self.routes: Optional[List[Route]] = None
        self.customer: Optional[Customer] = None
        self.customer_presence_time: float = 0

    def get_delay(self) -> float:
        delay: float = self.delay_mean
        if self.distribution.lower() == 'exp':
            delay = generate_exponential(self.delay_mean)
        elif self.distribution.lower() == 'generate_normal':
            delay = generate_normal(self.delay_mean, self.delay_dev)
        elif self.distribution.lower() == 'generate_uniform':
            delay = generate_uniform(self.delay_mean, self.delay_dev)
        return delay

    def get_next_element(self) -> Optional['Element']:
        if self.route_choice is None:
            return None

        if self.route_choice == 'probability':
            return choices([route.element for route in self.routes], 
                         [route.p for route in self.routes])[0]
        elif self.route_choice == 'priority':
            routes_by_queue = sorted(self.routes, key=lambda r: r.element.queue)
            return routes_by_queue[0].element

        raise ValueError('Make sure that either "probability" or "priority" is set as the route choice')

    def in_act(self, customer: Optional[Customer]) -> None:
        pass

    def out_act(self) -> None:
        self.quantity += 1
        if self.customer:
            self.customer_presence_time += (self.tcurr - self.customer.enter_time)

    def set_tcurr(self, tcurr: float) -> None:
        self.tcurr = tcurr

    def print_result(self) -> None:
        print(f'{self.name} quantity = {self.quantity}')

    def print_info(self) -> None:
        print(f'{self.name} state = {self.state} quantity = {self.quantity} tnext = {self.tnext}')

    def do_statistic(self, delta: float) -> None:
        pass

class Create(Element):
    def __init__(self, delay: float):
        super().__init__(delay=delay)
        self.tnext = 0.0
        self.failures: int = 0

    def out_act(self) -> None:
        super().out_act()
        self.tnext = self.tcurr + self.get_delay()
        customer = Customer(self.tcurr)
        route1, route2 = self.routes
        if route1.element.queue + route2.element.queue == route1.element.max_queue + route2.element.max_queue:
            self.failures += 1
        else:
            if route1.element.queue == route2.element.queue:
                next_element = random.choice([route1.element, route2.element])
            else:
                next_element = route1.element if route1.element.queue < route2.element.queue else route2.element
            next_element.in_act(customer)

class Process(Element):
    def __init__(self, delay: float, num_devices: int = 1):
        super().__init__(delay=delay)
        self.devices: List[Element] = [Element(delay=delay) for _ in range(num_devices)]
        for device in self.devices:
            device.tnext = float('inf')
        self.queue: int = 0
        self.max_queue: int = sys.maxsize
        self.mean_queue: float = 0.0
        self.tnext = float('inf')
        self.failure: int = 0
        self.average_load: float = 0
        self.aver_customers: float = 0

    def in_act(self, customer: Optional[Customer] = None) -> None:
        system_busy = True
        for device in self.devices:
            if device.state == 0:
                system_busy = False
                device.state = 1
                device.tnext = self.tcurr + self.get_delay()
                device.customer = customer
                break
        if system_busy:
            if self.queue < self.max_queue:
                self.queue += 1
        else:
            self.tnext = min(device.tnext for device in self.devices)

    def out_act(self) -> None:
        for device in self.devices:
            if self.tcurr >= device.tnext:
                super().out_act()
                device.out_act()
                device.tnext = float('inf')
                device.state = 0

                if self.queue > 0:
                    self.queue -= 1
                    device.state = 1
                    device.tnext = self.tcurr + self.get_delay()
                self.tnext = min(device.tnext for device in self.devices)

                next_route = super().get_next_element()
                if next_route is not None:
                    next_route.in_act(None)

    def is_available(self) -> bool:
        return any(device.state == 0 for device in self.devices)

    def print_info(self) -> None:
        for device in self.devices:
            device.print_info()
        print(f'failure = {self.failure}, queue = {self.queue}')

    def set_tcurr(self, tcurr: float) -> None:
        self.tcurr = tcurr
        for device in self.devices:
            device.tcurr = tcurr

    def do_statistic(self, delta: float) -> None:
        self.mean_queue += self.queue * delta
        self.average_load += delta * self.devices[0].state
        self.aver_customers += delta * (self.devices[0].state + self.queue)

class Model:
    def __init__(self, elements: List[Union[Create, Process]]):
        self.list: List[Union[Create, Process]] = elements
        self.tnext: float = 0.0
        self.event: int = 0
        self.tcurr: float = self.tnext
        self.queue_changes: int = 0

    def simulate(self, time: float) -> None:
        while self.tcurr < time:
            self.tnext = float('inf')
            for index, e in enumerate(self.list):
                if e.tnext < self.tnext:
                    self.tnext = e.tnext
                    self.event = index

            for e in self.list:
                e.do_statistic(self.tnext - self.tcurr)
            self.tcurr = self.tnext
            for e in self.list:
                e.set_tcurr(self.tcurr)
            self.list[self.event].out_act()
            for e in self.list:
                if e.tnext == self.tcurr:
                    e.out_act()
            self.check_queue_change()
        self.print_result()

    def check_queue_change(self) -> None:
        if self.list[1].queue - self.list[2].queue >= 2:
            self.list[1].queue -= 1
            self.list[2].queue += 1
            self.queue_changes += 1
        elif self.list[2].queue - self.list[1].queue >= 2:
            self.list[2].queue -= 1
            self.list[1].queue += 1
            self.queue_changes += 1

    def print_info(self) -> None:
        for e in self.list:
            e.print_info()

    def print_result(self) -> None:
        print('\n-------------RESULTS-------------')
        for e in self.list:
            e.print_result()
            if isinstance(e, Process):
                print(f'\taverage load = {e.average_load / self.tcurr}')
            print()

        print(f'lost customers percentage = {self.list[0].failures / self.list[0].quantity}')
        print(f'average customer time in bank = '
              f'{(self.list[1].devices[0].customer_presence_time + self.list[2].devices[0].customer_presence_time) / (self.list[1].quantity + self.list[2].quantity)}')

# Example usage
creator = Create(2.5)
processor1 = Process(1.5)
processor2 = Process(1.5)

creator.route_choice = 'priority'
creator.routes = [Route(processor1), Route(processor2)]

processor1.max_queue = 3
processor2.max_queue = 3

creator.name = 'Clients arrival'
processor1.name = 'Cashier 1'
processor2.name = 'Cashier 2'

creator.distribution = 'exp'
processor1.distribution = 'exp'
processor2.distribution = 'exp'

elements = [creator, processor1, processor2]
model = Model(elements)
model.simulate(1000)