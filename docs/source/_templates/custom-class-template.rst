{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members:

   {% block methods %}
   {% if methods %}
   {% set method_list = methods | list %}
   {% set non_private_methods_exist = namespace(value=false) %}
   {% for item in method_list %}
      {%- if not item.startswith('_') %}
         {% set non_private_methods_exist.value = true %}
      {%- endif %}
   {% endfor %}
   {% if non_private_methods_exist.value %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in method_list %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {% endfor %}
   {% endif %}
   {% endif %}
   {% endblock %}
