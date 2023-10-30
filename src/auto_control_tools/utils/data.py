from typing import Tuple, Union

import pandas as pd
from scipy.signal import lfilter

from .envoirment import is_jupyter_environment


class DataUtils:
    _jupyter_env = is_jupyter_environment()

    @staticmethod
    def linfilter(series: pd.Series, smothness: int) -> pd.Series:
        # the larger smothness is, the smoother curve will be
        b = [1.0 / smothness] * smothness
        a = 1
        return pd.Series(lfilter(b, a, series), name=series.name)

    @staticmethod
    def get_vreg(tf_data: pd.Series, settling_time_threshold: float = 0.02) -> Tuple[float, float]:
        for idx, value in tf_data.iloc[::1].items():
            local_s = tf_data[idx:]
            mean = local_s.mean()

            if all((local_s < (1 + settling_time_threshold) * mean) & (local_s > (1 - settling_time_threshold) * mean)):
                return idx, mean
        return 0, 0

    @staticmethod
    def get_max_tan(tf_data: pd.Series) -> Tuple[float, float]:
        diff = tf_data.diff()
        return float(diff.idxmax()), float(max(diff[1:]))

    @staticmethod
    def get_time_from_inclination(ref_time: float, ref_value: float, inclination: float, value: float) -> float:
        return (value - ref_value + inclination * ref_time) / inclination

    @classmethod
    def setup_data_default(cls, df: pd.DataFrame, sample_time: Union[float, None] = None,
                           step_signal: Union[float, None] = None,
                           use_lin_filter: bool = False, linfilter_sothness: int = 5
                           ) -> Tuple[pd.Series, float]:
        """
        Prepara os dados para uso dos métodos de identificação.

        Faz o preparo dos dados de :paramref:`df` adicionando campos informadas pelos parâmetros
        :paramref:`sample_time` e :paramref:`step_signal` e aplicando um filtro linear, caso solicitado.
        Retorna a pandas.Series referente ao sinal de saida em relação ao tempo e o valor do sinal degrau.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame com os dados de resposta a sinal degrau do sistema.
        sample_time : float, optional
            Valor do invervalo de amostragem. Caso informado, o intervalo de amostragem é considerado constante e
            igual ao valor fornecido.
        step_signal : float, optional
            Valor do sinal degrau de entrada. Se informado é considerado que o sinal está ativo em todos os momentos
            nos dados recebidos.
        use_lin_filter : bool, optional
            Em casos cujo sinal de saida é muito ruidoso, um filtro linear, para reduzir a oscilação dos dados é
            necessário para realizar a identificação. Se esse parâmetro for informado como verdadeiro, um filtro linear
            :meth:`DataUtils.linfilter` será aplicado sobre os dados.
        linfilter_sothness : bool, optional
            Valor inteiro referente a intensidade do filtro linear. Valores maiores reduzem mais o ruído, contudo
            tornam os dados mais distantes da realidade.

        Returns
        -------
        Retorna tupla com a pandas.Series referente ao sinal de saida em relação ao tempo e o valor do sinal degrau
        (informado em :paramref:`step_signal` ou obtido a partir dos dados).

        """
        if sample_time is not None:
            df['time'] = df.index * sample_time
        elif 'time' not in df.columns:
            raise ValueError('time column missing!')

        if step_signal is None:  # in case step signal is not informed, get it and then remove column
            step_signal = max(df['input'])
            df = cls.trunk_data_input(df)

        df = cls.offset_data_output(df)

        s = pd.Series(df['output'].values, index=df['time'])

        if use_lin_filter:
            s = DataUtils.linfilter(s, linfilter_sothness)

        return s, step_signal

    @classmethod
    def trunk_data_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove dados onde o valo de entrada é nulo.

        Recebe pandas.DataFrame com dados de resposta a sinal degrau de um sistema. Caso o sinal degrau inicie zerado
        e suba posteriormente, os momentos onde ele era igual a zero são removidos, deixando no DataFrame apenas
        dados onde o sinal degrau é ativo. Retorna o pandas.DataFrame com as alterações realizadas.
        """
        if 'input' not in df.columns:
            raise ValueError('input is not in df.columns')
        return df.loc[df['input'] != 0][[col for col in df if col != 'input']]

    @classmethod
    def offset_data_output(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove valores negativos e interpola os dados para encostarem no eixo :math:`t`

        Recebe pandas.DataFrame com dados de resposta a sinal degrau de um sistema, zera valores de saida
        abaixo de zero, e translada os dados de saida (sutrai um mesmo escalar de todos os valores de output) com
        base no valor de saida mais baixo presente, garantindo que a curva de saida encoste no eixo do tempo.
        Retorna o pandas.DataFrame com as alterações realizadas.
        """
        df['output'] = df['output'].apply(lambda x: x if x > 0 else 0)
        df['output'] = df['output'] - min(df['output'])
        return df
