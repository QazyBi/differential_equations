from abc import ABC, abstractmethod


class AbstractNumericalMethod(ABC):
    @abstractmethod
    def calculate_approximation(self, step):
        pass

    @abstractmethod
    def calculate_local_error(self):
        pass

    @abstractmethod
    def calculate_global_error(self):
        pass

    @abstractmethod
    def axis(self, step):
        pass

    @abstractmethod
    def axis_global_error(self):
        pass