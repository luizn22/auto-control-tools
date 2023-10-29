from abc import abstractmethod

from ..model import Model
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
            path, DataInputUtils.expected_fields(no_sample_time, no_step_signal), save_as=save_as)
