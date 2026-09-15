from abc import ABC, abstractmethod


class CocoSampleBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def register_sample(self):
        pass
    