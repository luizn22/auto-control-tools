import warnings
from typing import Tuple, Union

import pandas as pd
from scipy.signal import lfilter


class DataUtils:
    """
    Classe utilitária para manipulação de dados.

    Aqui são implementados diversos métodos para operação e extração de informações de
    :class:`pandas.Series` e :class:`pandas.DataFrame`.
    """

    @staticmethod
    def linfilter(series: pd.Series, smoothness: int) -> pd.Series:
        """
        Aplica um filtro linear à série temporal de entrada para suavização.

        Parameters
        ----------
        series : pandas.Series
            Dados da série temporal (:class:`pandas.Series`)
            de entrada a serem suavizados.
        smoothness : int
            Número inteiro que representa o nível de suavização. Quanto maior o valor, mais suave será a curva
            resultante.

        Returns
        -------
        pandas.Series
            Dados suavizados da série temporal.

        Notes
        -----
        A suavização é realizada usando o método `lfilter :func:`scipy.signal.lfilter` da biblioteca SciPy.

        Examples
        --------
        >>> dados_suavizados = act.DataUtils.linfilter(serie_entrada, 3)
        """
        # the larger smothness is, the smoother curve will be
        b = [1.0 / smoothness] * smoothness
        a = 1
        return pd.Series(lfilter(b, a, series), name=series.name)

    @staticmethod
    def get_vreg(tf_data: pd.Series, settling_time_threshold: float = 0.02) -> Tuple[float, float]:
        """
        Obtém o valor de regime de uma resposta a sinal degrau.

        Baseado em uma série temporal representativa da resposta de um :term:`Sistema` a um sinal degrau, obtém o valor
        de regime da resposta dentro do :paramref:`settling_time_threshold` especificado.

        Parameters
        ----------
        tf_data : pandas.Series
            Série temporal (:class:`pandas.Series`) representativa da resposta de um :term:`Sistema` a um sinal degrau.

        settling_time_threshold : float, optional
            Limiar de tempo de acomodação para determinar o valor de regime.
            O padrão é 0.02 (2%).

        Returns
        -------
        Tuple[float, float]
            Um tupla contendo o tempo de acomodação e o valor médio de regime.

        Examples
        --------
        >>> tempo_acomodacao, valor_regime = act.DataUtils.get_vreg(dados_resposta_degrau, 0.02)
        """
        for idx, value in tf_data.iloc[::1].items():
            local_s = tf_data.loc[tf_data.index >= idx]
            mean = local_s.mean()

            if all((local_s < (1 + settling_time_threshold) * mean) & (local_s > (1 - settling_time_threshold) * mean)):
                return idx, mean
        return 0, 0

    @staticmethod
    def get_max_tan(tf_data: pd.Series) -> Tuple[float, float, float]:
        """
        Obtém as coordenadas e o valor da inclinação do ponto de maior inclinação.

        Deriva os dados da série temporal representativa da resposta de um :term:`Sistema` a um sinal degrau e
        retorna as coordenadas (tempo e valor) e o valor da inclinação do ponto de maior inclinação.

        Parameters
        ----------
        tf_data : pandas.Series
            Série temporal (:class:`pandas.Series`) representativa da resposta de um :term:`Sistema` a um sinal degrau.

        Returns
        -------
        Tuple[float, float, float]
            Um tupla contendo as coordenadas (tempo e valor) do ponto de maior inclinação
            e o valor da inclinação nesse ponto.

        Examples
        --------
        >>> tempo, valor, inclinacao = act.DataUtils.get_max_tan(dados_resposta_degrau)
        """
        diff = tf_data.diff()
        idx_tan = float(diff.idxmax())
        return idx_tan, tf_data.loc[tf_data.index == idx_tan].iloc[0], float(max(diff[1:]))

    @staticmethod
    def get_time_at_value(reference_time: float, reference_value: float, slope: float, target_value: float) -> float:
        """
        Encontra o tempo para um valor de uma reta.

        Dados um ponto de referência (tempo e valor) pelo qual a reta de inclinação passa,
        a inclinação e um valor específico, calcula e retorna o tempo em que a reta passa por esse valor.

        Parameters
        ----------
        reference_time : float
            Tempo de referência pelo qual a reta de inclinação passa.
        reference_value : float
            Valor correspondente ao tempo de referência.
        slope : float
            Inclinação da reta.
        target_value : float
            Valor para o qual deseja-se encontrar o tempo correspondente.

        Returns
        -------
        float
            O valor correspondente ao tempo em que a reta passa por :paramref:`target_value`.

        Examples
        --------
        >>> tempo_correspondente = act.DataUtils.get_time_at_value(tempo_referencia, valor_referencia, inclinacao, valor_desejado)
        """
        return (target_value - reference_value + slope * reference_time) / slope

    @classmethod
    def setup_data_default(cls, df: pd.DataFrame, sample_time: Union[float, None] = None,
                           step_signal: Union[float, None] = None,
                           use_lin_filter: bool = False, linfilter_smoothness: int = 5
                           ) -> Tuple[pd.Series, float]:
        """
        Prepara os dados para uso dos métodos de :term:`Identificação`.

        Faz o preparo dos dados de :paramref:`df` adicionando campos informadas pelos parâmetros
        :paramref:`sample_time` e :paramref:`step_signal` e aplicando um filtro linear, caso solicitado.
        Retorna a pandas.Series referente ao sinal de saida em relação ao tempo e o valor do sinal degrau.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame com os dados de resposta a sinal degrau do :term:`Sistema`.
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
        linfilter_smoothness : bool, optional
            Valor inteiro referente a intensidade do filtro linear. Valores maiores reduzem mais o ruído, contudo
            tornam os dados mais distantes da realidade.

        Returns
        -------
        Tuple[pandas.Series, float]
            Retorna tupla com a :class:`pandas.Series`
            referente ao sinal de saida em relação ao tempo e o valor do sinal degrau (informado em
            :paramref:`step_signal` ou obtido a partir dos dados).

        """
        if sample_time is not None:
            df['time'] = df.index * sample_time
        elif 'time' not in df.columns:
            raise ValueError('time column missing!')

        if step_signal is None:  # in case step signal is not informed, get it and then remove column
            step_signal = max(df['input'])

        df, step_signal = cls.offset_data_input(df, step_signal)

        df = cls.trunk_data_input(df)

        df = cls.offset_data_output(df)

        s = pd.Series(df['output'].values, index=df['time'])

        if use_lin_filter:
            s = DataUtils.linfilter(s, linfilter_smoothness)

        noisy = cls.noise_check(s)

        if noisy:
            warnings.warn("Noisy data! This may cause problems. It might be a good idea to use the use_lin_filter "
                          "option.")

        return s, step_signal

    @classmethod
    def trunk_data_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove dados onde o valor de entrada é nulo.

        Recebe um :class:`pandas.DataFrame` com dados de
        resposta a um sinal degrau de um :term:`Sistema`. Caso o sinal degrau inicie zerado e suba posteriormente, os
        momentos onde ele era igual a zero são removidos, deixando no DataFrame apenas dados onde o sinal degrau é
        ativo. Retorna o pandas.DataFrame com as alterações realizadas.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame contendo dados de resposta a um sinal degrau de um :term:`Sistema`.

        Returns
        -------
        pandas.DataFrame
            DataFrame com os dados alterados, removendo os momentos onde o sinal degrau é igual a zero.

        Raises
        ------
        ValueError
            Se a coluna 'input' não estiver presente no DataFrame.

        Examples
        --------
        >>> df_alterado = act.DataUtils.trunk_data_input(dataframe_resposta_degrau)
        """
        if 'input' not in df.columns:
            raise ValueError('input is not in df.columns')
        df = df.loc[df['input'] > 0][[col for col in df if col != 'input']]
        df['time'] = df['time'] - df['time'].iloc[0]
        return df

    @staticmethod
    def offset_data_input(df: pd.DataFrame, step_signal: float) -> Tuple[pd.DataFrame, float]:
        if 'input' not in df.columns:
            raise ValueError('input is not in df.columns')

        pre_signal = df['input'].drop_duplicates().sort_values(ascending=True)
        if len(pre_signal) > 1:
            pre_signal = pre_signal.iloc[-2]
            step_signal = step_signal - pre_signal
        else:
            pre_signal = 0

        df['input'] = df['input'] - pre_signal
        return df, float(step_signal)

    @classmethod
    def offset_data_output(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove valores negativos e interpola os dados para encostarem no eixo do tempo.

        Recebe um :class:`pandas.DataFrame`
        com dados de resposta a um sinal degrau de um :term:`Sistema`, zera valores de saída abaixo de zero, e translada
        os dados de saída (subtrai um mesmo escalar de todos os valores de saída) com base no valor de saída mais baixo
        presente, garantindo que a curva de saída encoste no eixo do tempo. Retorna o pandas.DataFrame com as alterações
        realizadas.

        Parameters
        ----------
        df : pandas.DataFrame
            :class:`pandas.DataFrame` contendo dados de resposta a um sinal degrau de um :term:`Sistema`.

        Returns
        -------
        pandas.DataFrame
            DataFrame com os dados alterados, removendo valores negativos e garantindo que a curva de saída encoste no
            eixo do tempo.

        Notes
        -----
        A alteração dos valores negetivos para zero incorre na interpretação de um possível :term:`Sistema` com fase
        não mínima como apenas um :term:`Sistema` com atraso; isso pode ou não ser relevante dependendo do
        sistema específico.

        Examples
        --------
        >>> df_alterado = act.DataUtils.offset_data_output(dataframe_resposta_degrau)
        """
        df['output'] = df['output'].apply(lambda x: x if x > 0 else 0)
        df['output'] = df['output'] - min(df['output'])
        return df

    @staticmethod
    def noise_check(s: pd.Series, window_size: int = 5, deviation_percentage_threshold: float = 30) -> bool:
        """
        Verifica se a série temporal s possui ruido.

        Cria uma cópia suavizada da série e compara com a original para obter os ruídos locais, dividindo pelo
        valor da série suavizada para obter o desvio proporcional.
        Caso o desvio proporcional for maior que :paramref:`deviation_percentage_threshold` retorna verdadeiro.

        Parameters
        ----------
        s : pd.Series
            Série temporal a ser verificada.
        window_size: int
            Tamanho da janela de suavização. Padrão 5.
        deviation_percentage_threshold: float
            Desvio percentual de ruído aceito. Padrão (30%).

        Returns
        -------
        bool
            Se a série é ruidosa ou não

        """
        smoothed_series = s.rolling(window=window_size).mean()
        noise = (s - smoothed_series).abs() / smoothed_series

        noisy = any(noise.iloc[len(noise)//2:] > (deviation_percentage_threshold / 100))

        return noisy
