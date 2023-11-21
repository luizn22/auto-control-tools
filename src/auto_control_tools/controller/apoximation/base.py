from abc import abstractmethod

from ...model.model import Model
from ...controller.controller import Controller

P = 'P'
I = 'I'  # noqa
D = 'D'
PI = P+I
PD = P+D
PID = P+I+D


class BaseControllerAproximation:
    """
        Classe base para aproximação de controladores (:class:`Controller`).

        Métodos de aproximação de controladores devem ser subclasses desta classe. Elas devem implementar o método
        :meth:`get_controller` para retornar um objeto da classe :class:`Controller` com os ganhos referentes de
        :term:`Controlador PID`.

        Subclasses podem sobreescrever o atributo :attr:`_accepted_controllers`
        com os tipos de controlador suportados e utilizar o método
        :meth:`_parse_controller_option` para verifica se um tipo de controlador é aceito.
    """
    _accepted_controllers = [P, I, D, PI, PD, PID]

    @classmethod
    @abstractmethod
    def get_controller(cls, model: Model, controller_type: str) -> Controller:
        """
        Método abstrato :func:`abc.abstractmethod`
        para obtenção de um :class:`Controller`.
        """
        raise NotImplementedError('get_controller must be implemented in a subclass')

    @classmethod
    def _parse_controller_option(cls, controller_type: str):
        """Verifica se :paramref:`controller_type` está em :attr:`_accepted_controllers`, levanta uma exceção
        caso não esteja
        """
        if controller_type not in cls._accepted_controllers:
            raise ValueError(f'Invalid Controller option, must be one of {cls._accepted_controllers}')
