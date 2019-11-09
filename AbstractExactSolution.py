from abc import ABC, abstractmethod


class AbstractExactSolution(ABC):
    @abstractmethod
    def calculate_exact(self):
        pass

    @abstractmethod
    def y_prime_ith(self, i: float, y: float):
        pass

    @abstractmethod
    def exact_ith(self, i: float):
        pass
