"""auto_control_tools.analysis.stability

Implementação própria do critério de Routh–Hurwitz para sistemas contínuos.

Uso básico
----------
>>> from auto_control_tools.analysis.stability import RouthHurwitz, routh_hurwitz
>>> rh = RouthHurwitz([1, 2, 3, 4])  # s^3 + 2 s^2 + 3 s + 4
>>> result = rh.compute()
>>> result.is_stable
True
>>> result.rhp_poles
0

Projeto
-------
- Sem dependências externas obrigatórias (Pandas é opcional, usado apenas para
  organizar a tabela em DataFrame caso disponível).
- Algoritmo robusto com tratamento para:
  (a) pivô zero na primeira coluna (substituição por epsilon);
  (b) linha inteiramente nula (uso de polinômio auxiliar e derivada);
  (c) coeficientes de ordem ímpar/par; e
  (d) contagem de mudanças de sinal na primeira coluna.

Notas
-----
- A entrada são os coeficientes do polinômio característico em s, do termo de maior
  ordem para o termo independente (ex.: [1, 2, 3, 4] para s^3 + 2 s^2 + 3 s + 4).
- Os cálculos são feitos em ponto flutuante; se necessitar de exatidão simbólica,
  integrar posteriormente com SymPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


Number = float  # Alias simples; pode ser expandido para Decimal se necessário

# Exportações públicas deste módulo
__all__ = [
    "RouthHurwitz",
    "RouthResult",
    "RouthHurwitzError",
    "routh_hurwitz",
]


class RouthHurwitzError(Exception):
    """Erros específicos do módulo Routh–Hurwitz."""


@dataclass
class RouthResult:
    order: int
    coefficients: List[Number]
    table: List[List[Number]]
    row_labels: List[str]
    first_column: List[Number]
    rhp_poles: int
    is_stable: bool
    notes: List[str]

    def to_dataframe(self):
        """Retorna a tabela em um pandas.DataFrame, se pandas estiver disponível.
        Caso contrário, retorna None.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return None
        df = pd.DataFrame(self.table, index=self.row_labels)
        df.index.name = "Linha"
        return df

    def to_latex(self, floatfmt: str = ".6g") -> str:
        """Gera uma string LaTeX (tabular) da tabela de Routh.
        Parâmetros
        ----------
        floatfmt : str
            Formato numérico aplicado a cada elemento.
        """
        # Construção manual para não depender de pandas
        header_cols = max(len(r) for r in self.table)
        col_spec = "|" + "c|" * header_cols
        lines = ["\begin{tabular}{" + col_spec + "}", "\hline"]
        for row in self.table:
            padded = row + [0.0] * (header_cols - len(row))
            line = " & ".join(format(x, floatfmt) for x in padded)
            lines.append(line + " \ \hline")
        lines.append("\end{tabular}")
        return "".join(lines)


class RouthHurwitz:
    """Calcula a tabela de Routh–Hurwitz e a estabilidade de um polinômio.

    Parâmetros
    ----------
    coeffs : Sequence[Number]
        Coeficientes do polinômio característico em s, do termo de maior ordem
        para o termo independente. Ex.: ``[1, 2, 3, 4]`` representa
        ``s^3 + 2 s^2 + 3 s + 4``.
    epsilon : float, padrão 1e-6
        Valor pequeno usado quando um pivô da primeira coluna é zero, para
        evitar divisão por zero sem alterar o número de mudanças de sinal.
    zero_row_epsilon : float, padrão 1e-12
        Tolerância para decidir se uma linha é considerada "toda zero".
    normalize_leading : bool, padrão True
        Se True, normaliza o polinômio dividindo todos os coeficientes pelo
        coeficiente líder (a_n). Isso não altera a estabilidade.
    """

    def __init__(
        self,
        coeffs: Sequence[Number],
        *,
        epsilon: float = 1e-6,
        zero_row_epsilon: float = 1e-12,
        normalize_leading: bool = True,
    ) -> None:
        if not coeffs:
            raise RouthHurwitzError("A lista de coeficientes não pode ser vazia.")
        if all(abs(c) < zero_row_epsilon for c in coeffs):
            raise RouthHurwitzError("Todos os coeficientes são ~0. Polinômio inválido.")
        # Remove zeros à esquerda que possam ter sido passados por engano
        i = 0
        while i < len(coeffs) and abs(coeffs[i]) < zero_row_epsilon:
            i += 1
        if i == len(coeffs):
            raise RouthHurwitzError("Polinômio de grau indefinido (apenas zeros).")
        if i > 0:
            coeffs = coeffs[i:]
        # Normalização (opcional)
        if normalize_leading and abs(coeffs[0]) > 0:
            lead = float(coeffs[0])
            coeffs = [float(c) / lead for c in coeffs]
        else:
            coeffs = [float(c) for c in coeffs]

        self._coeffs = coeffs
        self._n = len(coeffs) - 1  # grau do polinômio
        self._eps = float(epsilon)
        self._zero_row_eps = float(zero_row_epsilon)

    # --------------------------- API pública --------------------------- #
    def compute(self) -> RouthResult:
        table, row_labels, notes = self._build_routh_table()
        first_col = [r[0] if r else 0.0 for r in table]
        rhp = self._count_sign_changes(first_col)
        return RouthResult(
            order=self._n,
            coefficients=list(self._coeffs),
            table=table,
            row_labels=row_labels,
            first_column=first_col,
            rhp_poles=rhp,
            is_stable=(rhp == 0),
            notes=notes,
        )

    # --------------------------- Implementação ------------------------ #
    def _build_routh_table(self) -> Tuple[List[List[Number]], List[str], List[str]]:
        n = self._n
        m_cols = (n + 2) // 2  # número de colunas
        row_labels = [f"s^{k}" for k in range(n, -1, -1)]
        notes: List[str] = []

        # Primeiras duas linhas a partir dos coeficientes
        row0 = [0.0] * m_cols
        row1 = [0.0] * m_cols
        # Mapeia coeficientes: a_n, a_{n-1}, ..., a_0
        # Linha s^n: a_n, a_{n-2}, a_{n-4}, ...
        # Linha s^{n-1}: a_{n-1}, a_{n-3}, a_{n-5}, ...
        for j in range(m_cols):
            k0 = 2 * j  # deslocamento par
            k1 = 2 * j + 1  # deslocamento ímpar
            a_even = self._coeff_at(n - k0)
            a_odd = self._coeff_at(n - k1)
            row0[j] = a_even
            row1[j] = a_odd
        table: List[List[Number]] = [row0, row1]

        # Itera para as demais linhas
        for i in range(2, n + 1):
            prev = table[i - 1]
            prev2 = table[i - 2]
            new_row = [0.0] * m_cols

            # Caso pivô zero
            if abs(prev[0]) < self._zero_row_eps:
                prev = prev.copy()
                prev[0] = self._eps
                notes.append(
                    f"Linha {row_labels[i-1]}: pivô 0 substituído por epsilon={self._eps:g}."
                )

            # Detecta linha toda zero (polinômio simétrico)
            if all(abs(x) < self._zero_row_eps for x in prev):
                # Polinômio auxiliar é formado a partir de prev2
                aux_poly_deg = n - (i - 2)  # grau do polinômio associado à linha prev2
                aux_coeffs = self._row_to_poly(prev2, aux_poly_deg)
                d_aux = self._poly_derivative(aux_coeffs)
                # Repreenche a linha i com os coeficientes de d_aux
                new_row = self._poly_to_row(d_aux, m_cols)
                notes.append(
                    f"Linha {row_labels[i-1]} toda zero: usada derivada do polinômio auxiliar."
                )
            else:
                # Fórmula padrão de Routh
                for j in range(m_cols - 1):
                    a = prev[0]
                    b = prev[j + 1]
                    c = prev2[0]
                    d = prev2[j + 1]
                    new_row[j] = ((a * d) - (c * b)) / a
                # Última coluna permanece 0.0 (ou já está)

            table.append(new_row)

        return table, row_labels, notes

    def _coeff_at(self, power: int) -> Number:
        """Retorna o coeficiente de s^power a partir de self._coeffs.
        self._coeffs = [a_n, a_{n-1}, ..., a_0]
        """
        idx_from_top = self._n - power
        if idx_from_top < 0 or idx_from_top >= len(self._coeffs):
            return 0.0
        return self._coeffs[idx_from_top]

    @staticmethod
    def _row_to_poly(row: List[Number], degree: int) -> List[Number]:
        """Converte uma linha de Routh (coef. intercalados) em polinômio completo.
        Ex.: para grau 4, linha [a4, a2, a0] -> [a4, 0, a2, 0, a0]
        """
        poly = [0.0] * (degree + 1)
        k = degree
        for x in row:
            if k < 0:
                break
            poly[degree - k] = x
            k -= 2
        return poly

    @staticmethod
    def _poly_to_row(poly: List[Number], m_cols: int) -> List[Number]:
        """Converte polinômio completo (grau n) para uma linha de Routh com m_cols.
        Pega coeficientes dos termos decrescendo de 2 em 2 (mesmo padrão das duas primeiras linhas).
        """
        n = len(poly) - 1
        row = [0.0] * m_cols
        j = 0
        for p in range(n, -1, -2):
            if j >= m_cols:
                break
            row[j] = poly[n - p]
            j += 1
        return row

    @staticmethod
    def _poly_derivative(poly: List[Number]) -> List[Number]:
        n = len(poly) - 1
        if n <= 0:
            return [0.0]
        # poly = [a_n, a_{n-1}, ..., a_0]
        deriv = [poly[i] * (n - i) for i in range(len(poly) - 1)]
        return deriv

    @staticmethod
    def _count_sign_changes(values: List[Number], tol: float = 1e-12) -> int:
        # Ignora zeros (|x| < tol) usando o sinal do menor epsilon positivo
        cleaned: List[float] = []
        for v in values:
            if abs(v) < tol:
                # substitui por um "quase zero" positivo para não criar artefato
                cleaned.append(1e-15)
            else:
                cleaned.append(v)
        changes = 0
        for a, b in zip(cleaned, cleaned[1:]):
            if a == 0 or b == 0:
                continue
            if (a > 0 and b < 0) or (a < 0 and b > 0):
                changes += 1
        return changes


def routh_hurwitz(coeffs: Sequence[Number], **kwargs) -> RouthResult:
    """Wrapper funcional: calcula Routh–Hurwitz e retorna `RouthResult`.
    Exemplos
    --------
    >>> routh_hurwitz([1, 2, 3, 4]).is_stable
    True
    """
    return RouthHurwitz(coeffs, **kwargs).compute()


# ------------------------- Execução direta (debug) ------------------------- #
if __name__ == "__main__":
    # Exemplos rápidos de verificação manual ("testes" smoke)
    examples = {
        "estavel: s^3 + 2s^2 + 3s + 4": [1, 2, 3, 4],     # estável
        "instavel: s^2 - 2s + 2": [1, -2, 2],              # 2 polos em RHP
        "marginal (linha nula): s^4 + 2s^2 + 1": [1, 0, 2, 0, 1],  # auxiliar/derivada
        "pivo zero: s^3 + 2s": [1, 0, 2, 0],               # força epsilon no pivô
        "estavel: s^2 + 2s + 2": [1, 2, 2],                # estável (−1 ± j)
    }
    for name, coeffs in examples.items():
        rh = RouthHurwitz(coeffs)
        res = rh.compute()
        print("Caso:", name)
        print("  Ordem:", res.order)
        print("  Primeira coluna:", [f"{x:.6g}" for x in res.first_column])
        print("  Polos em RHP:", res.rhp_poles, "| Estável:", res.is_stable)
        if res.notes:
            print("  Notas:", *res.notes, sep="- ")
        # Exibe tabela crua
        for lbl, row in zip(res.row_labels, res.table):
            print(" ", lbl, ":", [f"{x:.6g}" for x in row])
