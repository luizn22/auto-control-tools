from typing import Union

import control
import pandas as pd
import sympy as sp

from .model import Model


class FirstOrderModel(Model):
    """
    Classe para modelos clássicos de primeira ordem com atraso.

    Essa classe é voltada para casos específicos de modelos de primeira ordem que respeitam o formato:

    .. math::
        \\frac{K}{\\tau s + 1}e^{-\\theta s}

    Onde:

    * K é o ganho do sistema, representando a amplificação do sinal.
    * τ é uma constante de tempo, que indica o quão rápido o sistema responde a uma mudança.
    * θ é um termo relacionado ao atraso ou tempo morto no sistema.

    Essa forma específica da função de transferência é comumente usada para representar sistemas de primeira ordem
    com um atraso de tempo. Mas também pode ser usada para casos tem atraso, quando θ for zero.

    Sendo uma subclasse de :class:`Model`, essa classe adiciona suporte a definição de um modelo com os parâmetros
    K, tau e theta mas ainda mantém todas as funcionalidades da classe pai.

    Parameters
    ----------
    K : float
        Ganho do sistema.

        O termo K, se refere ao ganho do sistema, ele descreve a relação de amplificação entre a entrada e a saída de um
        sistema dinâmico. É um parâmetro crucial que determina a escala da resposta do sistema às mudanças na entrada.
    tau : float
        Constante de tempo de reação do sistema.

        A constante de tempo τ representa a velocidade com que o sistema atinge sua resposta estacionária ou valor de regime
        após uma perturbação. Em sistemas de primeira ordem, τ indica o tempo necessário para que a resposta atinja
        cerca de 63,2% de sua mudança total. Uma constante de tempo menor denota uma resposta mais rápida.
    theta : float
        Termo de atraso do sistema.

        O termo θ está associado a um atraso de tempo no sistema. Ele representa o tempo adicional que o sistema
        leva para responder a uma mudança na entrada, introduzindo uma componente de defasagem temporal na resposta.
    pade_degree : int, optional
        Grau do denominador de aproximação de atraso.

        Valor utilizado na aproximação do atributo :attr:`pade`, caso o sistema possua atraso (theta != 0).
        Quanto maior o grau mais próxima a aproximação, e consequentemente mais termos são adicionados a aproximação.

        Para mais detalhes sopre a aproximação do atraso verificar atributo :attr:`pade`.
    source_data : pd.Series, optional
        Conjunto de dados representando a variação da saida em relação ao tempo.
        Deve um objeto do tipo `Series <https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_
        da bilbioteca `pandas <https://pandas.pydata.org/docs/index.html>`_, sendo os valores representativos da saida
        e os valores de index representativos do tempo.

        Essa classe não faz nenhuma análise desses dados por si só, porém eles ficam salvos, e podem ser utilizados
        posteriormente, para plotagem de gráficos, por exemplo.

    Attributes
    ----------
    pade : control.TransferFunction
        Função de transferência representante do atraso do sistema.

        Caso haja atraso no sistema (theta != 0), é calculada a função de transferência que representa o termo de
        atraso:

        .. math:: e^{-\\theta s}

        A aproximação é feita através do método de Padé, utilizando a função `pade
        <https://python-control.readthedocs.io/en/latest/generated/control.matlab.pade.html>`_ da biblioteca de
        `controle <https://python-control.readthedocs.io/en/latest/index.html>`_, que resulta nos coeficientes de
        numerador e denominador de uma função de trasnferência que aproxima o atraso desejado.

        O parâmetro :paramref:`theta` representa o atraso de tempo, enquanto o parâmetro
        :paramref:`pade_degree` representa o grau do denominador da função de transferência utilizada para
        a aproximação.

    Notes
    -----
    O atributo :attr:`pade` é utilizado apenas para plotagem dos gráficos, visto que interfere nos cálculos de
    aproximação de controlador.

    Examples
    --------
    Uso básico:

    >>> model = FirstOrderModel(K=1, tau=2, theta=0.5)
    >>> model.view.print_tf()

    .. math::

        \\frac{1}{2s + 1}e^{-0.5s}

    Diferenças de :paramref:`pade_degree`:

    >>> model = FirstOrderModel(K=1, tau=2, theta=0.5, pade_degree=2)
    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/first_order_model_pade_deg_2.png


    >>> model = FirstOrderModel(K=1, tau=2, theta=0.5, pade_degree=10)
    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/first_order_model_pade_deg_10.png

    """
    def __init__(
            self,
            K: float,
            tau: float,
            theta: float = 0,
            pade_degree: int = 5,
            source_data: Union[pd.Series, None] = None,
    ):
        super().__init__(([K], [tau, 1]), source_data=source_data)
        self.K = K
        self.tau = tau
        self.theta = theta

        if theta != 0 and pade_degree > 0:
            self.pade = control.tf(*control.pade(theta, pade_degree))

            s = sp.symbols('s')
            self.tf_symbolic = self.tf_symbolic * sp.exp(-theta * s)
