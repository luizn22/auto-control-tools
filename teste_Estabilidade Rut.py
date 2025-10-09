"""
- numpy

Como usar
---------
1) Por equação (recomendado no TCC):
   s = np.poly1d([1, 0])
   den_expr = (s + 2) * (s + 3) * (s + 5)
   relatorio = analyze_and_report(den_expr)

2) Por coeficientes (ordem decrescente):
   den_coeffs = [1, 10, 35, 50]  # s^3 + 10 s^2 + 35 s + 50
   relatorio = analyze_and_report(den_coeffs)

Saída: relatório textual com a tabela de Routh, nº de polos no SPD e conclusão.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple


# =========================
# Utilidades de formatação
# =========================

def poly_to_str(coeffs: List[float], var: str = "s") -> str:
    """Converte uma lista de coeficientes (grau ↓) em string legível."""
    coeffs = np.array(coeffs, dtype=float)
    n = len(coeffs) - 1
    partes = []
    for i, c in enumerate(coeffs):
        p = n - i
        if abs(c) < 1e-12:
            continue
        sinal = " + " if c >= 0 else " - "
        v = abs(c)
        if p == 0:
            termo = f"{v:.6g}"
        elif p == 1:
            termo = f"{v:.6g}{var}"
        else:
            termo = f"{v:.6g}{var}^{p}"
        partes.append((sinal, termo))
    if not partes:
        return "0"
    s0, t0 = partes[0]
    s = ("" if s0.strip() == "+" else "- ") + t0
    for sgn, termo in partes[1:]:
        s += sgn + termo
    return s


def format_routh_table(table: np.ndarray) -> str:
    """Formata a tabela de Routh com rótulo de potência em cada linha."""
    rows, cols = table.shape
    linhas = []
    for i in range(rows):
        deg = (rows - 1) - i
        linha = f"s^{deg:>2} | " + "  ".join(f"{table[i, j]:>10.6g}" for j in range(cols))
        linhas.append(linha)
    return "\n".join(linhas)


# =========================
# Núcleo: Routh-Hurwitz
# =========================

def routh_table(den_coeffs: List[float], eps: float = 1e-9) -> Tuple[np.ndarray, Dict]:
    """
    Constrói a tabela de Routh-Hurwitz para um polinômio de denominador dado
    por coeficientes em ordem decrescente.

    Regras implementadas:
    - Substituição por ε quando o 1º elemento da linha é zero;
    - Tratamento de 'linha de zeros' via polinômio auxiliar (derivada).

    Retorna:
        table: ndarray (linhas = grau+1, colunas = floor(grau/2)+1)
        info:  dict com:
               - 'used_epsilon': linhas em que ε foi usado na 1ª coluna
               - 'row_of_zeros': índices de linhas que ficaram inteiramente zeradas
    """
    c = np.array(den_coeffs, dtype=float)

    # remove zeros à esquerda (se houver)
    nz = np.nonzero(np.abs(c) > 0)[0]
    if len(nz) == 0:
        raise ValueError("Polinômio nulo.")
    c = c[nz[0]:]

    n = len(c) - 1  # grau
    mcols = (n // 2) + 1
    table = np.zeros((n + 1, mcols), dtype=float)

    # Primeiras duas linhas
    table[0, :len(c[0::2])] = c[0::2]
    table[1, :len(c[1::2])] = c[1::2]

    used_eps = []
    zero_rows = []

    # Montagem das demais linhas
    for i in range(2, n + 1):
        # Caso especial: linha anterior toda zero -> usar polinômio auxiliar
        if np.allclose(table[i - 1, :], 0, atol=1e-12):
            zero_rows.append(i - 1)

            # Polinômio auxiliar E(s) vem da linha i-2 (termos de mesmo passo)
            order = n - (i - 2)
            row = table[i - 2, :]
            k = np.count_nonzero(~np.isclose(row, 0, atol=1e-12))
            row = row[:k]

            # Reconstrói E(s) com graus: order, order-2, order-4, ...
            degs = list(range(order, -1, -2))
            full = np.zeros(order + 1)
            for idx, deg in enumerate(degs):
                if idx < len(row):
                    full[order - deg] = row[idx]

            # Deriva E(s) e preenche a linha i-1 com graus: order-1, order-3, ...
            d = np.polyder(np.poly1d(full)).c  # graus decrescentes
            derived = []
            current_deg = len(d) - 1
            for deg in range(order - 1, -1, -2):
                idx = current_deg - deg
                derived.append(d[idx] if 0 <= idx < len(d) else 0.0)
            while len(derived) < mcols:
                derived.append(0.0)
            table[i - 1, :] = derived[:mcols]

        # Se o primeiro elemento da linha é zero, substitui por ε
        if abs(table[i - 1, 0]) < 1e-12:
            used_eps.append(i - 1)
            table[i - 1, 0] = eps

        # Regra de Routh para cada coluna
        for j in range(mcols - 1):
            a = table[i - 2, 0]
            b = table[i - 2, j + 1]
            c0 = table[i - 1, 0]
            d = table[i - 1, j + 1]
            table[i, j] = ((c0 * b) - (a * d)) / c0 if abs(c0) > 0 else 0.0

    info = {"used_epsilon": used_eps, "row_of_zeros": zero_rows}
    return table, info


def routh_stability(den) -> Dict:
    """
    Analisa a estabilidade via Routh-Hurwitz.

    Parâmetro:
        den: lista de coeficientes (grau ↓) ou np.poly1d

    Retorna:
        {
          'table': ndarray,
          'first_col': ndarray,
          'n_rhp' : nº de polos no semiplano direito,
          'strictly_stable': bool (todos polos em Re{s}<0),
          'imag_axis_roots': bool (há raiz(zes) em Re{s}=0),
          'info': {'used_epsilon': [...], 'row_of_zeros': [...]}
        }
    """
    den_coeffs = den.c.tolist() if isinstance(den, np.poly1d) else list(den)
    table, info = routh_table(den_coeffs)
    first = table[:, 0]

    # Mudanças de sinal na primeira coluna (ignorando zeros)
    signs = [np.sign(x) if abs(x) > 1e-8 else 0 for x in first]
    filtered = [s for s in signs if s != 0]
    n_changes = sum(1 for i in range(1, len(filtered)) if filtered[i] != filtered[i - 1])

    # Raízes no eixo imaginário?
    imag_axis = (len(info["row_of_zeros"]) > 0) \
                or np.any(np.isclose(first, 0, atol=1e-8)) \
                or (abs(den_coeffs[-1]) < 1e-12)  # termo constante zero -> raiz em s=0

    strictly_stable = (n_changes == 0) and (not imag_axis) and np.all(first > 0)

    return {
        "table": table,
        "first_col": first,
        "n_rhp": n_changes,
        "strictly_stable": bool(strictly_stable),
        "imag_axis_roots": bool(imag_axis),
        "info": info,
    }


def analyze_and_report(den) -> str:
    """
    Gera um relatório textual completo para o denominador informado.
    """
    den_coeffs = den.c.tolist() if isinstance(den, np.poly1d) else list(den)
    res = routh_stability(den_coeffs)

    txt = []
    txt.append("Polinômio (denominador): " + poly_to_str(den_coeffs, "s"))
    txt.append("\nTabela de Routh-Hurwitz:\n" + format_routh_table(res["table"]))

    if res["strictly_stable"]:
        txt.append("\nConclusão: ESTÁVEL (estritamente). Todos os polos no semiplano esquerdo.")
    elif res["imag_axis_roots"] and res["n_rhp"] == 0:
        txt.append("\nConclusão: MARGINALMENTE ESTÁVEL (há raiz(zes) no eixo imaginário; "
                   "resposta pode ter oscilações sustentadas/termo constante).")
    else:
        txt.append(f"\nConclusão: INSTÁVEL. Há {res['n_rhp']} polo(s) no semiplano direito.")

    # (Opcional) Checagem numérica por raízes:
    roots = np.roots(den_coeffs)
    roots_str = ", ".join(f"{r.real:.4g}{'+' if r.imag>=0 else '-'}j{abs(r.imag):.4g}" for r in roots)
    txt.append("\nRaízes (checagem numérica via numpy.roots): " + roots_str)

    # Interpretação prática curta:
    txt.append("\nInterpretação prática:")
    if res["strictly_stable"]:
        txt.append("- Resposta ao degrau é limitada e tende ao regime permanente (BIBO estável).")
        txt.append("- Todos os polos têm parte real negativa (Re{s}<0).")
    elif res["imag_axis_roots"] and res["n_rhp"] == 0:
        txt.append("- Sistema marginal: pode manter oscilações; pequeno atrito/controle altera o regime.")
        txt.append("- Em projeto, evite polos exatamente no eixo imaginário (robustez baixa).")
    else:
        txt.append("- Sistema instável: resposta diverge; requer realocação de polos (controle/realimentação).")
        if res["n_rhp"] == 1:
            txt.append("- Há 1 polo no semiplano direito (crescimento exponencial).")
        elif res["n_rhp"] > 1:
            txt.append(f"- Há {res['n_rhp']} polos no semiplano direito (divergência mais severa).")

    return "\n".join(txt)


# =========================
# Auto-testes (didáticos)
# =========================

def run_self_tests() -> None:
    """
    Roda 3 casos clássicos e verifica a coerência dos resultados.
    """
    s = np.poly1d([1, 0])

    # 1) ESTÁVEL: (s+2)(s+3)(s+5)
    den1 = (s + 2) * (s + 3) * (s + 5)
    r1 = routh_stability(den1)
    assert r1["strictly_stable"] and r1["n_rhp"] == 0 and not r1["imag_axis_roots"]

    # 2) INSTÁVEL: s^2 - 1 -> raízes em +1 e -1 (1 polo no SPD)
    den2 = [1, 0, -1]
    r2 = routh_stability(den2)
    assert (not r2["strictly_stable"]) and (r2["n_rhp"] == 1) and (not r2["imag_axis_roots"])

    # 3) MARGINAL: s^2 + 1 -> raízes ±j
    den3 = [1, 0, 1]
    r3 = routh_stability(den3)
    assert (not r3["strictly_stable"]) and (r3["n_rhp"] == 0) and r3["imag_axis_roots"]

    print("Auto-testes: OK (estável / instável / marginal).")


# =========================
# Execução direta (exemplos)
# =========================
if __name__ == "__main__":
    # ---- Definição por EQUAÇÃO (edite aqui) ----
    s = np.poly1d([1, 0])
    den_expr = (s + 2) * (s + 3) * (s + 5)   # estável
    # den_expr = s**2 - 1                     # instável (troque a linha para testar)
    # den_expr = s**2 + 1                     # marginal (troque a linha para testar)

    print("=== ANÁLISE (por equação) ===")
    print(analyze_and_report(den_expr))

    # ---- Exemplo por COEFICIENTES (opcional) ----
    # den_coeffs = [1, 10, 35, 50]
    # print("\n=== ANÁLISE (por coeficientes) ===")
    # print(analyze_and_report(den_coeffs))

    # ---- Auto-testes didáticos ----
    print("\n=== AUTO-TESTES ===")
    run_self_tests()
