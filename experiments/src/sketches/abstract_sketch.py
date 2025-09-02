from abc import ABC, abstractmethod

class Sketch(ABC):
    
    @abstractmethod
    def update(self, x: int, d: int = 1) -> None:
        pass  # This is an abstract method, no implementation here.
    
    @abstractmethod
    def estimate(self, x: int, tau: float | None) -> int:
        pass 