*********
Glossário
*********

.. glossary::

    Sistema
        Um sistema, no contexto de controle de processos, é descrito como um objeto ou conjunto de objetos que visa
        alcançar um objetivo específico, cujas propriedades são estudadas. Exemplos incluem sistemas de fabricação,
        circuitos elétricos, sistemas biológicos, entre outros. Este termo abrange uma variedade de entidades,
        desde processos industriais até sistemas biológicos e econômicos :cite:`CoelhoIdentificacao`.

    Função de Transferência
        A função de transferência de um :term:`Sistema` representado por uma equação diferencial linear invariante no
        tempo é definida como a relação entre a transformada de Laplace da saída (função de resposta — response
        function) e a transformada de Laplace da entrada (função de excitação — driving function), admitindo-se todas as
        condições iniciais nulas :cite:`ogata2010engenharia`.

    Modelo
        Em controle de processos, um modelo refere-se a uma representação matemática abstrata de um :term:`Sistema`,
        capturando seus aspectos essenciais para análise, simulação e tomada de decisões. Geralmente, o modelo é
        expresso por equações que descrevem a dinâmica do sistema. Na prática, busca-se um modelo que seja adequado
        para uma aplicação específica de controle, priorizando utilidade sobre a exatidão absoluta
        :cite:`CoelhoIdentificacao`.

    Identificação
        No contexto de controle de processos, a identificação refere-se ao processo de determinar o :term:`Modelo`
        matemático de um sistema a partir de dados observados. Envolve técnicas e procedimentos para extrair informações
        relevantes sobre a dinâmica do :term:`Sistema`, permitindo a criação de um modelo representativo.
        A identificação é crucial para adaptar o modelo às características específicas do sistema real, garantindo sua
        eficácia em aplicações práticas, como diagnóstico, supervisão, otimização e controle
        :cite:`CoelhoIdentificacao`.

    Malha Aberta
        A malha aberta refere-se a sistemas de controle em que a saída não influencia diretamente o comportamento do
        :term:`Sistema`. Nesses casos, a saída não é medida nem utilizada para ajustes em tempo real,
        resultando em operações baseadas em sequências pré-determinadas, como em uma máquina de lavar roupas.
        Embora sejam mais simples e menos dispendiosos, os sistemas de malha aberta enfrentam desafios relacionados
        à necessidade de calibração precisa e à suscetibilidade a distúrbios, exigindo ajustes periódicos para manter a
        qualidade da saída desejada :cite:`ogata2010engenharia`.

    Sintonia
        No contexto de controle de sistemas, a sintonia refere-se ao processo de ajuste dos parâmetros de um
        controlador para otimizar o desempenho do :term:`Sistema`. Essa atividade é essencial para alcançar respostas
        rápidas e estáveis, minimizando erros e garantindo uma operação eficiente em :term:`Malha Fechada`.
        A sintonia pode ser realizada através de métodos experimentais, como testes de :term:`Malha Aberta`, ou por
        abordagens analíticas baseadas em modelos matemáticos. O objetivo é encontrar uma configuração ideal que
        maximize a eficácia do controlador em atender às demandas específicas do sistema, resultando em um comportamento
        preciso e responsivo :cite:`apostpidsint`.

    Malha Fechada
        A malha fechada, ou sistema de controle com realimentação, é um paradigma em que a saída de um :term:`Sistema` é
        comparada à entrada de referência, utilizando a diferença para ajustar e controlar o sistema. Este método,
        ilustrado pelo controle de temperatura em um ambiente, permite uma resposta mais estável e insensibilidade a
        distúrbios externos. Em contraste, sistemas de :term:`Malha Aberta` não utilizam a saída como meio de controle,
        dependendo da calibração prévia e enfrentando desafios em termos de precisão e adaptação a distúrbios
        imprevistos :cite:`ogata2010engenharia`.

    Controlador PID
        O controlador PID, ou Proporcional-Integral-Derivativo, é uma técnica amplamente utilizada em sistemas de
        controle em :term:`Malha Fechada` para manter variáveis em um valor desejado. Esse tipo de controlador é
        composto por três termos principais:

        - proporcional, que ajusta a saída proporcionalmente ao erro atual;
        - integral, que acumula o erro ao longo do tempo e corrige o viés sistemático;
        - e derivativo, que prevê tendência futura do erro, permitindo uma resposta mais rápida a alterações repentinas.

        Esses três componentes trabalham em conjunto para proporcionar estabilidade, precisão e resposta dinâmica em uma
        variedade de sistemas. O controlador PID é uma ferramenta fundamental no campo de controle de sistemas
        dinâmicos, contribuindo para otimizar o desempenho e a eficiência em diversas aplicações
        :cite:`ogata2010engenharia`.

    Aproximação de Ganhos
        No contexto da :term:`Sintonia` eficaz de :term:`Controlador PID`, busca-se uma aproximação precisa dos
        parâmetros para otimizar o desempenho em sistemas de controle em :term:`Malha Fechada`. Métodos experimentais
        de :term:`Malha Aberta`, como a aplicação de um sinal degrau na saída do controlador e o subsequente registro
        da resposta do :term:`Sistema`, são comumente empregados para essa finalidade. Essa abordagem, independente de
        métodos específicos, visa aprimorar a determinação dos parâmetros PID, facilitando a sintonia fina do
        controlador e, consequentemente, melhorando a eficiência do sistema :cite:`apostpidsint`.
