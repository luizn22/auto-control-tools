from abc import abstractmethod
from typing import Dict

from .base import BaseControllerAproximation
from ...model.first_order_model import FirstOrderModel
from ...controller.controller import Controller


class FirstOrderTableControllerAproximationItem:
    """
    Classe de itens de :class:`FirstOrderTableControllerAproximation`.

    Utilizada para facilitar a implementação de "Métodos de Tabela" conforme descrito na classe
    :class:`FirstOrderTableControllerAproximation`.

    Para implementar subclasse:

    - Definir controller_type;
    - Sobreescrever :meth:`get_controller` que recebe um modelo :class:`FirstOrderModel` e retorna um controlador
      :class:`Controller`;
    """
    controller_type: str = ''  # replace with controller type

    @classmethod
    @abstractmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        """
        `Método abstrato :func:`abc.abstractmethod`
        para obtenção de um :class:`Controller` baseado em um :class:`FirstOrderModel`.
        """
        raise NotImplementedError('get_controller must be implemented in a subclass')


class FirstOrderTableControllerAproximation(BaseControllerAproximation):
    """
    Classe base para métodos de aproximação de primeira ordem baseados em tabelas.

    Diversos métodos de aproximação de controlador utilizam tabelas dependentes apenas nos valores do :term:`Modelo`
    paramétrico caracterizado pela seguinte :term:`Função de Transferência` :footcite:p:`CoelhoChapter4`:

    .. math::
        \\frac{K}{\\tau s + 1}e^{-\\theta s}

    A classe :class:`FirstOrderModel` é especializada nesse tipo de :term:`Modelo`.

    Esta classe visa facilitar a implementação destes comumente referidos como "Métodos de Tabela", pois possuem
    formulas simples e espessíficas para o ganho de cada parâmetro de PID, a depender do controlador desejado.

    Para implementar um controlador como subclasse de :class:`FirstOrderTableControllerAproximation`,
    devem ser seguidos os seguinte passos:

    - Adicionar a classe :class:`FirstOrderTableControllerAproximation` a herança da nova classe;
    - Declarar o :attr:`_accepted_controllers` com os tipos de controladores aceitos;
    - Declarar o dicionário :attr:`_controller_table` com os tipos de controladores como chaves e referências a
      subclasses de :class:`FirstOrderTableControllerAproximationItem` como valores;
    - Por fim, basta implementar cada uma das classes derivadas de :class:`FirstOrderTableControllerAproximationItem`.

    Examples
    --------

    Exemplos de implementação são as classes :class:`ZieglerNicholsControllerAproximation` e
    :class:`CohenCoonControllerAproximation`.

    """
    _controller_table: Dict[str, FirstOrderTableControllerAproximationItem] = {}
    _accepts_null_theta = True

    @classmethod
    def get_controller(
            cls,
            model: FirstOrderModel,  # type: ignore[override]
            controller_type: str
    ) -> Controller:
        """
        Manuseador dos métodos de aproximação da classe.

        Verifica se controller_type é permitido, então chama o método
        :meth:`FirstOrderTableControllerAproximationItem.get_controller` apropriado.

        Parameters
        ----------
        model : FirstOrderModel
            :term:`Modelo` para o qual deve ser feita a aproximação.
        controller_type : str
            Tipo de controlador desejado.

        Returns
        -------
        Controlador com os parâmetros encontrados
        """
        if cls._accepts_null_theta is False and model.theta <= 0:
            raise ValueError('This method does not work with theta <= 0!')

        cls._parse_controller_option(controller_type)
        return cls._controller_table[controller_type].get_controller(model)
