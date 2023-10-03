# auto-control-tools

![Tests](https://github.com/luizn22/auto-control-tools/actions/workflows/tests.yml/badge.svg)

## Useful scripts:
- install requirements: `pip install -r requirements.txt`, `pip install -r requirements_dev.txt`
- install project as package: `pip install -e .`
- run typing check: `mypy src`
- run format check: `flake8 src`
- run tests: `pytest`
- build docs locally: `sphinx-build -b html .\docs\source\ .\rtd_build\`