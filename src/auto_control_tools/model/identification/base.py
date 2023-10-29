from abc import abstractmethod
from copy import copy
from typing import List, Tuple, Union

import pandas as pd

from ..model import Model
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class BaseModelIdentification:
    """
        Classe base para identificação de modelos (:class:`Model`).

        Métodos de identificação de Modelos devem ser subclasses desta classe. Elas devem implementar o método
        :meth:`get_model` para retornar um objeto da classe :class:`Model` que represente o modelo matemático
        do sistema que produziu os dados recebidos.

        Em geral, implementações farão a identificação com base nos dados de resposta a sinal degrau. Para tanto,
        o método :meth:`get_data_input_layout` oferece um leiaute para serem informados os dados referentes a
        resposta do sistema a um sinal degrau em relação ao tempo. Implementações de :meth:`get_model` podem ler os
        dados e aplicar seus métodos de identificação específicos.

        Notes
        -----
        Alguns métodos privados foram criados para facilitar a implementação de novas classes de identificação.
        A documentação deles pode ser vista na base de código.

    """
    @classmethod
    @abstractmethod
    def get_model(cls, *args, **kwargs) -> Model:
        """
        `Método abstrato <https://docs.python.org/3/library/abc.html#abc.abstractmethod>`_
        para obtenção de um :class:`Model`.
        """
        raise NotImplementedError('get_model must be implemented in a subclass')

    @classmethod
    def _get_model_data_default(cls, path: str, sample_time: Union[None, float] = None,
                                step_signal: Union[float, None] = None) -> pd.DataFrame:
        """
        Obtém os campos esperados, lê os dados do arquivo indicado através de
        :meth:`DataInputUtils.read_table_with_fields` verifica se não existem campos faltantes
        e retorna o pandas.DataFrame resultante.

        Parameters
        ----------
        path : str
            Caminho até o arquivo a ser lido. O leiaute pode ser obtido através de :meth:`get_data_input_layout`.

        sample_time : float, optional
            Valor do invervalo de amostragem. Caso informado, o intervalo de amostragem é considerado constante e
            igual ao valor fornecido.

        step_signal : float, optional
            Valor do sinal degrau de entrada. Se informado é considerado que o sinal está ativo em todos os momentos
            nos dados recebidos.

        Returns
        -------
        pandas.Dataframe os dados do arquivo recebido e as colunas esperadas
        """
        expected_fields = cls._expected_fields(sample_time, step_signal)
        df = DataInputUtils.read_table_with_fields(path, expected_fields)

        if any(f not in df.columns for f in expected_fields):
            missing_fields = [f for f in expected_fields if f not in df.columns]
            raise ValueError(f'The fields {missing_fields} are required and were informed in the input data')

        return df

    @classmethod
    def _setup_data_default(cls, df: pd.DataFrame, sample_time: Union[float, None] = None,
                            step_signal: Union[float, None] = None,
                            use_lin_filter: bool = False, linfilter_sothness: int = 5
                            ) -> Tuple[pd.Series, float]:
        """
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
            df['time'] = df.index * step_signal

        if step_signal is None:  # in case step signal is not informed, get it and then remove column
            step_signal = max(df['input'])
            df = cls._trunk_data_input(df)

        df = cls._offset_data_output(df)

        s = pd.Series(df['output'].values, index=df['time'])

        if use_lin_filter:
            s = DataUtils.linfilter(s, linfilter_sothness)

        return s, step_signal

    @classmethod
    def _offset_data_output(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe pandas.DataFrame com dados de resposta a sinal degrau de um sistema, zera valores de saida
        abaixo de zero, e translada os dados de saida (sutrai um mesmo escalar de todos os valores de output) com
        base no valor de saida mais baixo presente, garantindo que a curva de saida encoste no eixo do tempo.
        Retorna o pandas.DataFrame com as alterações realizadas.
        """
        df['output'] = df['output'].apply(lambda x: x if x > 0 else 0)
        df['output'] = df['output'] - min(df['output'])
        return df

    @classmethod
    def _trunk_data_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe pandas.DataFrame com dados de resposta a sinal degrau de um sistema. Caso o sinal degrau inicie zerado
        e suba posteriormente, os momentos onde ele era igual a zero são removidos, deixando no DataFrame apenas
        dados onde o sinal degrau é ativo. Retorna o pandas.DataFrame com as alterações realizadas.
        """
        if 'input' not in df.columns:
            raise ValueError('input is not in df.columns')
        return df.loc[df['input'] != 0][[col for col in df if col != 'input']]

    @classmethod
    def get_data_input_layout(cls, path: str, save_as: str = 'xlsx', no_sample_time: bool = False,
                              no_step_signal: bool = False):
        """
        Salva tabela com leiaute de dados de entrada esperado no caminho especificado.
        
        Nome do arquivo gerado data_input.csv ou data_input.xlsx

        Para realizar a Identificação de um :class:`Model` são necessários dados referentes a resposta a sinal degrau
        da planta. Esse método visa facilitar o processo fornecendo um arquivo (Planilha de Excel ou arquivo CSV) com
        os cabeçalhos esperados para ser preenchido com os dados do sistema.

        Uma vez com os dados inseridos no arquivo, o método :meth:`get_model` pode ser chamado, passando o caminho do
        aquivo no parâmetro :paramref:`get_model.path`.

        Alguns parâmetros como sample_time e step signal podem simplificar a entrada de dados.

        Parameters
        ----------
        path : str
            Caminho até a pasta onde deve ser salvo o arquivo de leiaute.
        save_as : str
            Tipo de arquivo do leiaute.

            .. list-table:: Tipos aceitos
                :header-rows: 1

                * - String de entrada
                  - Tipo de arquivo
                  - Terminação
                * - xlsx
                  - Planilha do Excel
                  - .xlsx
                * - csv
                  - CSV
                  - .csv

        no_sample_time : float, optional
            Caso verdadeiro a coluna de tempo não é adicionada no leiaute. Nesse caso o valor do tempo
            de amostragem (*sample_time*) deverá ser informado ao fornecer os dados ao método de identificação.

        no_step_signal : float, optional
            Caso verdadeiro a coluna de sinal degrau não é adicionada no leiaute. Nesse caso o valor do sinal degrau
            (*step_signal*) deverá ser informado ao fornecer os dados ao método de identificação.

        """
        DataInputUtils.create_table_with_fields(
            path, cls._expected_fields(no_sample_time, no_step_signal), save_as=save_as)

    @classmethod
    def _expected_fields(cls, sample_time: Union[float, bool, None] = None,
                         step_signal: Union[float, bool, None] = None) -> List[str]:
        """
        Retorna a lista DataInputUtils.standard_fields com exeção dos campos sample_time ou step_signal caso sejam
        informados nos parâmetros.
        """
        fields = copy(DataInputUtils.standard_fields)

        if sample_time is not None:
            fields.remove('time')

        if step_signal is not None:
            fields.remove('input')

        return fields
