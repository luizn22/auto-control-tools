import os
from typing import List, Union
from copy import copy

import pandas as pd


class DataInputUtils:
    """
    Classe utilitária para a entrada de dados.

    Aqui são implementados diversos métodos para criação e leitura de tabelas referentes a entrada dos dados utilizados
    para :term:`Identificação` de plantas de sistemas de controle.

    Attributes
    ----------

    default_table_name : str
        Nome padrão para as tabelas criadas, default: ``'data_input'``
    standard_fields : List[str]
        Campos padrão para as tabelas criadas default: ``['time', 'input', 'output']``.
    allowed_file_type : List[str]
        Tipos de arquivos de tabela permitidos e suportados: ``['csv', 'xlsx']``.

    """

    default_table_name = 'data_input'  # update BaseModelIdentification.get_data_input_layout docstring if changed
    standard_fields = ['time', 'input', 'output']
    allowed_file_type = ['csv', 'xlsx']

    @classmethod
    def create_table_with_fields(
            cls,
            path: str,
            fields: List[str],
            table_name: str = default_table_name,
            save_as: str = 'csv'
    ) -> str:
        """
        Cria um arquivo contendo uma tabela vazia com os campos especificados.

        Cria um arquivo (CSV ou Excel) contendo uma tabela vazia com os campos especificados.
        O arquivo é salvo no caminho fornecido e seu nome é determinado pelo parâmetro :paramref:`table_name`.

        Parameters
        ----------
        path : str
            Caminho onde o arquivo será salvo.
        fields : List[str]
            Lista dos campos da tabela.
        table_name : str, optional
            Nome da tabela. O padrão é ``'data_input'``.
        save_as : str, optional
            Formato do arquivo a ser salvo (``'csv'`` ou ``'xlsx'``). O padrão é ``'csv'``.

        Returns
        -------
        str
            Caminho completo para o arquivo criado.

        Raises
        ------
        ValueError
            Se o formato de arquivo fornecido não estiver entre os permitidos.

        Examples
        --------
        >>> file_path = act.DataInputUtils.create_table_with_fields('/path/to/save', ['time', 'input', 'output'])
        """
        if save_as not in cls.allowed_file_type:
            raise ValueError(f'{save_as} is not an allowed file type')

        file_path = os.path.join(path, f'{table_name}.{save_as}')
        df = pd.DataFrame(columns=fields)

        if save_as == 'csv':
            df.to_csv(file_path, index=False)
        elif save_as == 'xlsx':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f'{save_as} is not an allowed file type')

        return file_path

    @classmethod
    def read_table_with_fields(cls, path: str, fields: Union[List[str], None] = None) -> pd.DataFrame:
        """
        Lê um arquivo contendo uma tabela e retorna um
        :class:`pandas.DataFrame` com **apenas** os campos
        especificados.

        Lê um arquivo (CSV ou Excel) contendo uma tabela e retorna um
        :class:`pandas.DataFrame` com apenas
        os campos especificados.
        O caminho do arquivo e os campos desejados são fornecidos como parâmetros.

        Parameters
        ----------
        path : str
            Caminho do arquivo a ser lido.
        fields : List[str], optional
            Lista dos campos desejados. Se None, utiliza os campos padrão da classe.
            O padrão é ``['time', 'input', 'output']``.

        Returns
        -------
        pandas.DataFrame
            :class:`pandas.DataFrame`
            contendo os dados da tabela e os campos especificados.

        Raises
        ------
        ValueError
            Se o formato de arquivo fornecido não estiver entre os permitidos.

        Examples
        --------
        >>> df = act.DataInputUtils.read_table_with_fields('/path/to/table.xlsx', ['input', 'output'])
        """
        if fields is None:
            fields = cls.standard_fields

        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError(f'{os.path.splitext(path)[-1]} is not an allowed file type')

        df = df[[f for f in fields if f in df.columns]]

        return df

    @classmethod
    def expected_fields(cls, sample_time: Union[float, bool, None] = None,
                        step_signal: Union[float, bool, None] = None) -> List[str]:
        """
        Retorna os campos esperados.

        Retorna a lista :attr:`standard_fields` com exeção dos campos :paramref:`sample_time` ou
        :paramref:`step_signal` caso sejam
        informados nos parâmetros.
        """
        fields = copy(cls.standard_fields)

        if sample_time is not None:
            fields.remove('time')

        if step_signal is not None:
            fields.remove('input')

        return fields

    @classmethod
    def get_model_data_default(cls, path: str, sample_time: Union[None, float] = None,
                               step_signal: Union[float, None] = None) -> pd.DataFrame:
        """
        Salva planilha de leiaute para preenchimento.

        Obtém os campos esperados, lê os dados do arquivo indicado através de
        :meth:`DataInputUtils.read_table_with_fields` verifica se não existem campos faltantes
        e retorna o pandas.DataFrame resultante.

        Parameters
        ----------
        path : str
            Caminho até o arquivo a ser lido. O leiaute pode ser obtido através de
            :meth:`~BaseModelIdentification.get_data_input_layout`.

        sample_time : float, optional
            Valor do invervalo de amostragem. Caso informado, o intervalo de amostragem é considerado constante e
            igual ao valor fornecido.

        step_signal : float, optional
            Valor do sinal degrau de entrada. Se informado é considerado que o sinal está ativo em todos os momentos
            nos dados recebidos.

        Returns
        -------
        pandas.DataFrame
            os dados do arquivo recebido e as colunas esperadas.
        """
        expected_fields = cls.expected_fields(sample_time, step_signal)
        df = cls.read_table_with_fields(path, expected_fields)

        if any(f not in df.columns for f in expected_fields):
            missing_fields = [f for f in expected_fields if f not in df.columns]
            raise ValueError(f'The fields {missing_fields} are required and were not informed in the input data')

        return df
