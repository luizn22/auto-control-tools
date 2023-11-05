.. _class-ref:

.. currentmodule:: auto_control_tools

*********************
Referência de Classes
*********************

A Biblioteca em Python de Sistemas de Controle :mod:`auto_control_tools` proporciona classes e métodos para
:term:`Identificação` e :term:`Aproximação de Ganhos` de :term:`Controlador PID` de sistemas de controle em malha
fechada.

A documentação está disponível em duas formas: docstrings providenciadas junto ao código, e um guia de usuário,
disponível em `<https://auto-control-tools.readthedocs.io/pt-br/latest/>`_.

Os exemplos das docstrings assumem os seguintes comandos de importação
    >>> import control
    >>> import auto_control_tools as act

Modelo
======

Classes de modelo, representativas do :term:`Modelo` matemático de uma planta de sistemas de controle.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

    Model
    ModelView
    FirstOrderModel

.. _identif:

Métodos de Identificação
========================

Classes de :term:`Identificação` são subclasses de :class:`BaseModelIdentification`, e implementam o método
:meth:`~BaseModelIdentification.get_model`, que faz a :term:`Identificação` de dados discretos de resposta a sinal
degrau de um sistema de controle e retorna objeto da classe :class:`Model` ou de uma de suas subclasses.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

    ZieglerNicholsModelIdentification
    HagglundModelIdentification
    SmithModelIdentification
    SundaresanKrishnaswamyModelIdentification
    NishikawaModelIdentification
    BaseModelIdentification


Controlador
===========

Classes de controlador, representativas de um sistema em :term:`Malha Fechada` ocorrendo o controle de um
:term:`Modelo` através de um :term:`Controlador PID`.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

    Controller
    ControllerView


.. _gain-aprox:

Métodos de Aproximação de Ganhos
================================

Classes de Aproximação de Controlador são subclasses de :class:`BaseControllerAproximation`, e implementam o método
:meth:`~BaseControllerAproximation.get_controller`, que faz a :term:`Aproximação de Ganhos` de um
:term:`Controlador PID` para um objeto de :term:`Modelo` da classe :class:`Model` ou uma de suas subclasses.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

    ZieglerNicholsControllerAproximation
    CohenCoonControllerAproximation
    BaseControllerAproximation
    FirstOrderTableControllerAproximation
    FirstOrderTableControllerAproximationItem


Visualização de Dados
=====================

Classes de Visualização se dados são utilitárias a outras classes da biblioteca.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

    PlotUtils


Manipulação de Dados
=====================

Classes de Manipulação se dados são utilitárias a outras classes da biblioteca.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

    DataInputUtils
    DataUtils

