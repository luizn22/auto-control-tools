from abc import abstractmethod

from ..model import Model


class BaseModelIdentification:
    @abstractmethod
    def get_model(self, *args, **kwargs) -> Model:
        raise NotImplementedError('get_model must be implemented in a subclass')

    @abstractmethod
    def get_data_input_layout(self, path: str, *args, **kwargs):
        raise NotImplementedError('get_data_input_layout must be implemented in a subclass')


