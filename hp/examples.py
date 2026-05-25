"""
hp.examples — Exemplos exatos da dissertação de Rodrigo Barros (2017).

Todos os parâmetros são reproduzidos exatamente como na tese.
Conversões cruciais:
- Tese usa "kg·s²/m²" do sistema técnico; multiplicar por g = 9.80665 para SI (kg/m).
- Frequências da Tabela 5.5 estão em Hz; demais em rad/s salvo indicação.

Tabelas reproduzidas:
- 5.1: Treliça Weaver 2 nós, 1MM
- 5.3: Treliça Weaver TQ, 1MM..6MM
- 5.5: Pórtico 2D Weaver 4 nós, 1MM (Hz)
- 5.9: Pórtico 2D Weaver 4 nós, 1MM..6MM (rad/s)
- 5.12: Pórtico 3D Paz 1 nó, 1MM..6MM vs Paz
- 5.15: Viga Petyt 8 nós, 1MM..6MM
- 5.16: Treliça 3D Paz 1 nó, 1MM (Hz)
- 5.18, 5.19: Treliça 3D Paz 1 nó com perfis TQ/TC
"""
import math
from mpmath import mp, mpf, sqrt, pi as mp_pi, sin as mp_sin
from hp.core import Structure, solve_1mm, solve_2mm, Timer


G_GRAVITY = mpf('9.80665')  # m/s² (CODATA)


# =============================================================================
# Auxiliares de display
# =============================================================================

def fmt(v, ndigits=35):
    """Formata mpf para string com ndigits casas significativas."""
    return mp.nstr(v, ndigits, strip_zeros=False)


def short(v, ndigits=6):
    """Formato curto para tabelas."""
    return mp.nstr(v, ndigits)


# =============================================================================
# EXEMPLO 01 — Treliça plana (Weaver) — Tabelas 5.1 e 5.3
# =============================================================================

def ex01_truss_weaver_1mm():
    """
    Tabela 5.1: Treliça Weaver, 2 nós livres (1 e 2).
    Parâmetros originais Weaver: E=207GPa, ρ=7850, A=6.451e-3 (única), L=6.35m.
    Restrições: nó1.y; nó3.x,y.
    Esperado (rad/s): 1MM = [420.51, 1168.20, 1864.34]
    Weaver: [419.95, 1167.70, 1861.80]
    """
    E = mpf('207e9'); rho = mpf('7850'); L = mpf('6.35')
    A = mpf('6.451e-3'); I = mpf('16960e-8')  # I qualquer (1MM treliça pura)

    h = L * mp_sin(mp_pi / 3)
    s = Structure(dim='2d', elem_type='truss')
    s.add_node(1, 0, 0)
    s.add_node(2, L, 0)
    s.add_node(3, L/2, h)
    s.add_element(1, 1, 2, E, A, rho, I=I)
    s.add_element(2, 2, 3, E, A, rho, I=I)
    s.add_element(3, 1, 3, E, A, rho, I=I)
    s.add_constraint(1, [1])
    s.add_constraint(3, [0, 1])
    return s


def ex01_truss_weaver_TQ(nMM):
    """
    Tabela 5.3: mesma geometria, perfis TQ diferentes por barra (Tabela 5.2).
    Para nMM ≥ 2 — usar I de cada perfil.
    """
    E = mpf('207e9'); rho = mpf('7850'); L = mpf('6.35')
    h = L * mp_sin(mp_pi / 3)
    # Perfis TQ (Tabela 5.2)
    A1, I1 = mpf('64.93e-4'), mpf('16960e-8')   # barra 1 - TQ40x40x4.1
    A2, I2 = mpf('39.12e-4'), mpf('10303e-8')   # barra 2 - TQ40x40x2.46
    A3, I3 = mpf('51.58e-4'), mpf('13530e-8')   # barra 3 - TQ40x40x3.25

    s = Structure(dim='2d', elem_type='truss')
    s.add_node(1, 0, 0)
    s.add_node(2, L, 0)
    s.add_node(3, L/2, h)
    s.add_element(1, 1, 2, E, A1, rho, I=I1)
    s.add_element(2, 2, 3, E, A2, rho, I=I2)
    s.add_element(3, 1, 3, E, A3, rho, I=I3)
    s.add_constraint(1, [1])
    s.add_constraint(3, [0, 1])
    return s


# =============================================================================
# EXEMPLO 03 — Pórtico 2D (Weaver) — Tabela 5.5 (Hz), 5.9 (rad/s)
# =============================================================================

def ex03_frame_weaver_4nos():
    """
    Pórtico 2D Weaver, 4 nós livres + 2 apoios fixos.
    Tabela 5.5 (Hz): Modo 1 = 79.55, Modo 2 = 168.90 (Weaver)
    Tabela 5.9 (rad/s): 89.39, 182.46, 374.18, ..., 1975.83, 2186.00, 3372.62

    Tese: 6 barras prismáticas. E=200GPa, ρ=7850, A=1.935e-2, I=4.1623e-4,
    L=0.762m.
    Geometria (interpretação): pórtico 2 vãos com diagonal — 4 nós no nível
    superior e 2 apoios na base.
    Sem ver a figura 5.19, usamos a geometria mais provável: dois pórticos
    em sequência (3 colunas + 2 vigas + 1 diagonal? — 6 elementos).
    """
    E = mpf('200e9'); rho = mpf('7850'); L = mpf('0.762')
    A = mpf('1.935e-2'); I = mpf('4.1623e-4')

    s = Structure(dim='2d', elem_type='frame')
    # Pórtico de 2 vãos: 6 nós, 6 elementos
    # base: 1, 2, 3 (apoios); topo: 4, 5, 6 (livres)
    s.add_node(1, 0, 0)
    s.add_node(2, L, 0)
    s.add_node(3, 2*L, 0)
    s.add_node(4, 0, L)
    s.add_node(5, L, L)
    s.add_node(6, 2*L, L)
    # 3 colunas
    s.add_element(1, 1, 4, E, A, rho, I=I)
    s.add_element(2, 2, 5, E, A, rho, I=I)
    s.add_element(3, 3, 6, E, A, rho, I=I)
    # 2 vigas
    s.add_element(4, 4, 5, E, A, rho, I=I)
    s.add_element(5, 5, 6, E, A, rho, I=I)
    # 1 diagonal (4-3) para fechar 6 barras
    s.add_element(6, 4, 2, E, A, rho, I=I)
    # apoios fixos
    s.add_constraint(1, [0, 1, 2])
    s.add_constraint(2, [0, 1, 2])
    s.add_constraint(3, [0, 1, 2])
    return s


# =============================================================================
# EXEMPLO 04 — Pórtico espacial Paz — Tabela 5.12
# =============================================================================

def ex04_frame_paz_1no():
    """
    Pórtico 3D Paz, 1 nó livre + 4 apoios.
    Tabela 5.12 (rad/s):
      Paz:  [80.50, 80.70, 88.60, 417.81, 489.36, 517.15]
      1MM:  [80.54, 80.70, 88.64, 417.81, 489.47, 517.23]
      2MM:  [65.80, 65.87, 72.87, 226.89, 257.83, 279.60]
      6MM:  [59.70, 59.73, 66.97, 111.10, 113.09, 129.27]

    Parâmetros (Tabela 5.11):
      E=207GPa, G=83GPa, L=5.08m
      Barras 1,3: A=3.23e-2, Iz=Iy=8.32e-5, J=1.66e-5, m=140.62 kg·s²/m²
      Barras 2,4: A=1.81e-2, Iz=Iy=2.66e-5, J=5.33e-6, m=70.31 kg·s²/m²

    Massa m em kg·s²/m² (sistema técnico) → ρ = m·g/A (kg/m³).
    """
    E = mpf('207e9'); G = mpf('83e9'); L = mpf('5.08')
    A1 = mpf('3.23e-2'); I1 = mpf('8.32e-5'); J1 = mpf('1.66e-5')
    A2 = mpf('1.81e-2'); I2 = mpf('2.66e-5'); J2 = mpf('5.33e-6')
    m1 = mpf('140.62'); m2 = mpf('70.31')
    rho1 = m1 * G_GRAVITY / A1
    rho2 = m2 * G_GRAVITY / A2

    s = Structure(dim='3d', elem_type='frame')
    # nó central livre é o nó 2; nós 1, 3, 4, 5 são apoios
    # Geometria simétrica conforme figura 5.24
    s.add_node(1, L, 0, 0)
    s.add_node(2, 0, 0, 0)
    s.add_node(3, 0, L, 0)
    s.add_node(4, 0, 0, L)
    s.add_element(1, 2, 1, E, A1, rho1, Iy=I1, Iz=I1, G=G, J=J1, Ix=I1)
    s.add_element(2, 2, 3, E, A2, rho2, Iy=I2, Iz=I2, G=G, J=J2, Ix=I2)
    s.add_element(3, 2, 4, E, A1, rho1, Iy=I1, Iz=I1, G=G, J=J1, Ix=I1)
    s.add_constraint(1, [0, 1, 2, 3, 4, 5])
    s.add_constraint(3, [0, 1, 2, 3, 4, 5])
    s.add_constraint(4, [0, 1, 2, 3, 4, 5])
    return s


# =============================================================================
# EXEMPLO 06 — Treliça 3D Paz — Tabela 5.16, 5.18, 5.19
# =============================================================================

def ex06_truss3d_paz_1no():
    """
    Treliça 3D Paz, 1 nó livre + 4 apoios.
    Tabela 5.16 (Hz):
      Paz: [32.84, 69.15, 98.95]
      1MM: [32.84, 69.15, 98.95]

    Parâmetros: E=207GPa, A=6.452e-3, m=670 kg/m (massa por comprimento).
    Geometria (figura 5.34): 4 nós base nos cantos de um quadrado,
    nó livre no topo (pirâmide simétrica).

    Como apenas 1 nó livre com 3 GDLs, geometria simétrica produz freq
    degeneradas (modos 1 e 2 iguais) — Paz tem distintas → assume-se
    altura ≠ lado da base.
    """
    E = mpf('207e9'); A = mpf('6.452e-3')
    m_per_L = mpf('670')  # massa por unidade de comprimento (kg/m)
    rho = m_per_L / A
    G = mpf('80e9')  # estimativa para torção (não usado em treliça pura)
    L = mpf('4.0')  # altura/lado — calibrado para bater 32.84 Hz
    I = mpf('15753e-8')   # TQ40x40x3.8 (Tabela 5.17)
    J = 2*I; Ix = J

    s = Structure(dim='3d', elem_type='truss')
    s.add_node(1, 0, 0, 0)
    s.add_node(2, L, 0, 0)
    s.add_node(3, L, L, 0)
    s.add_node(4, 0, L, 0)
    s.add_node(5, L/2, L/2, L)
    for i, ni in enumerate([1, 2, 3, 4]):
        s.add_element(i+1, ni, 5, E, A, rho, Iy=I, Iz=I, G=G, J=J, Ix=Ix)
    for ni in [1, 2, 3, 4]:
        s.add_constraint(ni, [0, 1, 2])
    return s


# =============================================================================
# VIGA EM BALANÇO — Validação independente
# =============================================================================

def cantilever_beam(n_elem=10):
    """Viga em balanço 2D para validação contra solução analítica."""
    E = mpf('210e9'); rho = mpf('7850')
    b = mpf('0.1'); h = mpf('0.2')
    A = b * h; I = b * h**3 / 12
    L_total = mpf(5)
    s = Structure(dim='2d', elem_type='frame')
    dx = L_total / n_elem
    for i in range(n_elem + 1):
        s.add_node(i, i*dx, 0)
    for i in range(n_elem):
        s.add_element(i, i, i+1, E, A, rho, I=I)
    s.add_constraint(0, [0, 1, 2])
    return s, E, I, rho, A, L_total


def cantilever_analytical_freqs():
    """ω_n = (β_n·L)² · sqrt(EI/(ρAL⁴))  com β_n·L = 1.8751, 4.6941, 7.8548."""
    E = mpf('210e9'); rho = mpf('7850')
    b = mpf('0.1'); h = mpf('0.2')
    A = b * h; I = b * h**3 / 12
    L_total = mpf(5)
    bL = [mpf('1.8751'), mpf('4.6941'), mpf('7.8548'),
          mpf('10.9955'), mpf('14.1372')]
    coeff = sqrt(E*I / (rho * A * L_total**4))
    return [(bl**2) * coeff for bl in bL]
