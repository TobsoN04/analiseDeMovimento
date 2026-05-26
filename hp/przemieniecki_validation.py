"""
hp.przemieniecki_validation — Validação contra J. S. Przemieniecki,
"Theory of Matrix Structural Analysis" (McGraw-Hill, 1968).

Esse livro é o pai do método dos elementos finitos matricial moderno e tem
exemplos clássicos perfeitos para validar nosso solver nMM.

Exemplos validados:
- Cap 12.5: Barra fixo-livre, 2 elementos longitudinais
  - 1MM: ω₁L·√(ρ/E) = 1.6114, ω₂L·√(ρ/E) = 5.6293
  - Solução analítica: ω₁L·√(ρ/E) = π/2 ≈ 1.5708; ω₂L·√(ρ/E) = 3π/2 ≈ 4.7124
- Eq. 10.81: m₀ da treliça 3D = (ρAL/6)·[[2I₃, I₃],[I₃, 2I₃]]  ✓
- Eq. 10.110: m₂ da treliça axial (com fator 2 vs nossa convenção)
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from mpmath import mp, mpf, sqrt, pi as mp_pi
mp.dps = 60

from hp.core import Structure, Timer
from hp.nmm_solver import solve_struct_nmm


# Valores tabulados de Przemieniecki para barra fixo-livre 2 elementos (Cap 12.5)
PRZEM_2ELEM = {
    'w1_normalized': mpf('1.6114'),  # ω₁·L·√(ρ/E) com L = comprimento de UM elemento
    'w2_normalized': mpf('5.6293'),
}


def fixed_free_bar(n_elem, total_length=mpf(1)):
    """
    Barra axial fixo-livre, n_elem elementos iguais.
    Comprimento total = total_length.
    Parâmetros: E=1, ρ=1, A=1 (normalizados).
    """
    E = mpf(1); rho = mpf(1); A = mpf(1)
    dL = total_length / n_elem
    s = Structure(dim='2d', elem_type='truss')
    for i in range(n_elem + 1):
        s.add_node(i, i * dL, 0)
    for i in range(n_elem):
        s.add_element(i, i, i+1, E, A, rho, I=mpf('1e-10'))
    # Fixar nó 0 (extremidade fixa) e direção y (não há flexão real aqui)
    s.add_constraint(0, [0, 1])
    for i in range(1, n_elem + 1):
        s.add_constraint(i, [1])  # impedir movimento transversal espúrio
    return s


def analytical_fixed_free_freqs(L=mpf(1), E=mpf(1), rho=mpf(1), n_modes=6):
    """
    Solução analítica: u(x,t) = sin((2n-1)πx/(2L))·cos(ωₙt)
    ωₙ = (2n-1)π/(2L)·√(E/ρ)
    """
    return [(2*n - 1) * mp_pi / (2*L) * sqrt(E/rho) for n in range(1, n_modes + 1)]


def validate_fixed_free():
    """Replica Cap 12.5 e estende com mais elementos / nMM."""
    print('=' * 110)
    print('  VALIDAÇÃO PRZEMIENIECKI Cap 12.5 — Barra fixo-livre (vibração longitudinal)')
    print('=' * 110)
    print(f'  Parâmetros: E=1, ρ=1, A=1, L_total=2 (normalizados)')
    print()

    L_total = mpf(2)
    f_anal = analytical_fixed_free_freqs(L=L_total, n_modes=6)
    print(f'  Solução analítica (ωₙ = (2n-1)π/(2L)·√(E/ρ)):')
    for i in range(min(3, len(f_anal))):
        print(f'    ω{i+1} = {mp.nstr(f_anal[i], 25)}')

    print(f'\n  Przemieniecki 1MM com 2 elem (normalizado por L_elem=1):')
    # Tabela 12.5: 1.6114·√(E/ρ)/L_elem com L_elem=1, então ω₁ = 1.6114
    # Mas L_elem = L_total/2 = 1, então ω₁·2 = 1.6114 → ω₁ = 0.8057
    print(f'    ω₁ = 1.6114/(L_elem)·√(E/ρ) com L_elem=1: 1.6114 → /2 = {mpf("0.8057")}')
    print(f'    ω₂ = 5.6293/(L_elem)·√(E/ρ): 5.6293 → /2 = {mpf("2.81465")}')

    print(f'\n  {"n_elem":>6} | {"nMM":>3} | {"ω₁":>22} | {"erro vs anal":>12} | {"ω₂":>22} | {"erro vs anal":>12} | {"t (s)":>6}')
    print('  ' + '-' * 110)
    for n_elem in [2, 4, 8]:
        s = fixed_free_bar(n_elem, total_length=L_total)
        for n_mm in [1, 2, 3, 4]:
            try:
                with Timer() as t:
                    omegas, _ = solve_struct_nmm(s, n_mm)
                if len(omegas) >= 2:
                    e1 = abs(omegas[0] - f_anal[0]) / f_anal[0] * 100
                    e2 = abs(omegas[1] - f_anal[1]) / f_anal[1] * 100
                    print(f'  {n_elem:>6} | {n_mm:>3} | {mp.nstr(omegas[0], 17):>22} | '
                          f'{mp.nstr(e1, 6):>11}% | {mp.nstr(omegas[1], 17):>22} | '
                          f'{mp.nstr(e2, 6):>11}% | {float(t.elapsed):>5.2f}')
                else:
                    print(f'  {n_elem:>6} | {n_mm:>3} |   (apenas 1 modo)')
            except Exception as ex:
                print(f'  {n_elem:>6} | {n_mm:>3} |   ERRO: {ex}')

    print()


def validate_m0_consistency():
    """Validar que nosso M1 da treliça bate com Eq. (10.81) de Przemieniecki."""
    print('=' * 110)
    print('  VALIDAÇÃO Eq. (10.81) — Mass matrix consistente da barra de treliça')
    print('=' * 110)
    from hp.core import truss_1d_K0_M1
    E = mpf(1); A = mpf(1); rho = mpf(1); L = mpf(1)
    K0, M1 = truss_1d_K0_M1(E, A, rho, L)

    # Przemieniecki Eq. (10.81): m₀ = (ρAL/6)·[[2,1],[1,2]] (para 2D barra)
    m0_przem = (rho * A * L / mpf(6)) * mp.matrix([[2, 1], [1, 2]])
    print(f'  Nossa M1[0,0] = {mp.nstr(M1[0,0], 30)}')
    print(f'  Przemieniecki m₀[0,0] = {mp.nstr(m0_przem[0,0], 30)}')
    print(f'  Diff: {mp.nstr(abs(M1[0,0] - m0_przem[0,0]), 8)}')
    print()


def validate_m2_consistency():
    """Validar nossa M2 vs m₂ Eq. (10.110) de Przemieniecki (com fator 2 de convenção)."""
    print('=' * 110)
    print('  VALIDAÇÃO Eq. (10.110) — m₂ da barra (Przemieniecki vs nossa M2)')
    print('=' * 110)
    from hp.core import truss_1d_M2
    E = mpf(1); A = mpf(1); rho = mpf(1); L = mpf(1)
    M2_ours = truss_1d_M2(E, A, rho, L)

    # Przemieniecki Eq. (10.110): m₂ = (2ρ²Al³/(45E)) · [[1, 7/8],[7/8, 1]]
    # = ρ²AL³/(180E) · [[8, 7],[7, 8]]
    m2_przem = rho**2 * A * L**3 / (mpf(180) * E) * mp.matrix([[8, 7], [7, 8]])

    print(f'  Nossa M2[0,0] = {mp.nstr(M2_ours[0,0], 25)}')
    print(f'  Przem m₂[0,0] = {mp.nstr(m2_przem[0,0], 25)}')
    print(f'  Razão Przem/Nossa = {mp.nstr(m2_przem[0,0] / M2_ours[0,0], 10)}')
    print()
    print(f'  Esperado: razão = 2 (convenção: Przemieniecki define m = m₀ + ω²m₂ + ω⁴m₄,')
    print(f'                    nós definimos K = K₀ - ω²M₁ - ω⁴M₂)')
    print(f'                    A relação fica m_przem(2n) = 2 * M_(n)_tese para n>=1.')
    print(f'                    (diferença advém de ω²·m(ω) expandido vs K(ω) direto)')


def run_all():
    t_g = Timer()
    with t_g:
        validate_m0_consistency()
        validate_m2_consistency()
        validate_fixed_free()
    print(f'\n{"=" * 110}')
    print(f'  TEMPO TOTAL Przemieniecki: {float(t_g.elapsed):.2f} s · precisão {mp.dps} dígitos')
    print('=' * 110)


if __name__ == '__main__':
    run_all()
