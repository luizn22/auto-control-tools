Ferramentas de Análise Clássica
================================

Esta seção reúne as principais ferramentas de análise adicionadas à
biblioteca ``auto_control_tools``, com foco em ensino de Sistemas de
Controle:

- Discretização contínuo–discreto (ZOH, FOH, Tustin, Forward Euler…)
- Resposta impulsiva com métricas temporais completas
- Mapa de polos e zeros no plano complexo
- Critério de estabilidade de Routh–Hurwitz

Discretização de Sistemas
-------------------------

.. automodule:: auto_control_tools.analysis.discretization
   :members: Discretizer, DiscretizationResult, DiscretizationMethod
   :undoc-members:
   :show-inheritance:

Resposta Impulsiva
------------------

.. automodule:: auto_control_tools.response.impulse
   :members: Impulse, ImpulseView, impulse_response, impulse_analysis
   :undoc-members:
   :show-inheritance:

Mapa de Polos e Zeros
---------------------

.. automodule:: auto_control_tools.analysis.poles
   :members: PoleZeroAnalyzer, PoleZeroPlotConfig, pzmap
   :undoc-members:
   :show-inheritance:

Critério de Routh–Hurwitz
--------------------------

.. automodule:: auto_control_tools.analysis.stability
   :members: RouthHurwitz, RouthResult, routh_hurwitz
   :undoc-members:
   :show-inheritance:
