from typing import Union, List, Dict, Any, Tuple

import control
import pandas as pd
import sympy as sp

from ..utils.data import DataUtils
from ..utils.plot import PlotUtils


class Model:
    """
    Classe representativa do modelo matemático de uma planta.

    Típicamente o modelo matemático de uma planta no domínio da frequenica pode ser definido por um numerador e um
    denominador em potências de s. Ex:

    .. math::

        G(s) = \\frac{ s + 1 }{ s^2 + s + 1 }


    Essa classe funciona guardando um objeto de função de transferência (:attr:`tf`) da biblioteca de sistemas de
    controle do python (`control <https://python-control.readthedocs.io/en/latest/index.html>`_), representando o modelo
    matemático de uma planta no domínio da frequenica, além de guardar alguns outros metadados sobre a função de
    transferência, que poderão ser utilizados para identificação de modelo posteriormente, por exemplo.


    Além disso o atributo :attr:`view` possibilita a visualização de dados, estatísticas e gráficos referentes a
    função de trasnferência.


    Parameters
    ----------
    tf : ~control.TransferFunction, tuple[list[float], list[float]]]
        Função de transferência que representa o processo da planta,
        pode ser tanto as listas dos coeficientes,
        quanto com um objeto de função de transferência da bilbiotéca de controle
        (`TransferFunction
        <https://python-control.readthedocs.io/en/latest/generated/control.TransferFunction.html>`_).
        Ex:

        .. code-block:: python

            num = [1, 2, 3]
            den = [4, 5, 6, 7]
            model = Model((num, den))
            model.view.print_tf()
            >>

        .. math::

            \\frac{ s^2 + 2s + 3 }{ 4s^3 + 5s^2 + 6s + 7 }

        .. code-block:: python

            num = [1, 2, 3]
            den = [1, 1, 1, 0]
            tf = control.TransferFunction(num, den)
            model = Model(tf)
            model.view.print_tf()
            >>

        .. math::

            \\frac{ s^2 + 2s + 3 }{ s^3 + s^2 + s }

    source_data : [pd.Series, optional]
        Conjunto de dados representando a variação da saida em relação ao tempo.
        Deve um objeto do tipo `Series <https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_
        da bilbioteca `pandas <https://pandas.pydata.org/docs/index.html>`_, sendo os valores representativos da saida
        e os valores de index representativos do tempo.

        Essa classe não faz nenhuma análise desses dados por si só, porém eles ficam salvos, e podem ser utilizados
        posteriormente, para plotagem de gráficos, por exemplo.

    Attributes
    ----------
    tf : control.TransferFunction
        Função de transferência que representa o processo da planta (`TransferFunction
        <https://python-control.readthedocs.io/en/latest/generated/control.TransferFunction.html>`_).

        Utilizada pelas classes de visualização de dados (:class:`ModelView` e :class:`ControllerView`) e pelas
        classes de aproximação de modelo para aproximação de ganhos (:ref:`gain-aprox`).

    view : ModelView
        Utilizado para visualização dos dados, estatísticas e gráficos referentes ao Modelo.

        Verificar a classe :class:`ModelView` para os métodos de visualização de dados.
    """
    def __init__(
            self,
            tf: Union[control.TransferFunction, Tuple[List[float], List[float]]],
            source_data: Union[pd.Series, None] = None
    ):
        """Instancia Modelo matemático de uma planta"""
        if isinstance(tf, control.TransferFunction):
            self.tf = tf
        else:
            self.tf = control.TransferFunction(*tf)

        self.num = self.tf.num[0][0].tolist()
        self.den = self.tf.den[0][0].tolist()

        s = sp.symbols('s')
        num_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.num))])
        den_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.den))])
        self.tf_symbolic = num_symbolic / den_symbolic

        self.order = self._identify_model_order()
        self.source_data = source_data if source_data is not None else pd.Series().astype(float)

        self.pade = None

        self.view: ModelView = ModelView(self)

    def _identify_model_order(self) -> int:
        return int(self.tf.den[0][0].size) - 1


class ModelView:
    def __init__(self, model: Model):
        self.model = model

    def plot_model_graph(self):
        if self.model.source_data.empty:
            PlotUtils.plot_tf(self.model.tf, pade=self.model.pade)
        else:
            PlotUtils.plot_tf(self.model.tf, self.model.source_data, pade=self.model.pade)

    def get_model_data(self) -> Dict[str, Any]:
        return dict(control.step_info(self.model.tf))

    def print_model_data(self, *args, **kwargs):
        DataUtils.pprint_dict(self.get_model_data())

    def print_tf(self):
        PlotUtils.print_tf(self.model.tf_symbolic)
