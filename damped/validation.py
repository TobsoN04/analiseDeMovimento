"""
damped.validation — Comparação versão amortecida vs não-amortecida.

Para cada exemplo, calcula:
- ω_n (frequência natural não-amortecida)
- ω_d (frequência amortecida = ω_n·sqrt(1-ζ²))
- ζ (razão de amortecimento)
- shift relativo Δω/ω_n
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from mpmath import mp, mpf, sqrt
from hp.core import Timer, solve_1mm
from hp.examples import (
    ex01_truss_weaver_TQ, ex03_frame_weaver_4nos,
    ex04_frame_paz_1no, ex06_truss3d_paz_1no,
    cantilever_beam, short, fmt,
)
from damped.core import (
    rayleigh_damping, solve_quadratic_eigenvalue,
    damped_modal_frequencies,
)


def run_example(name, struct, zeta1=mpf('0.02'), zeta2=mpf('0.02')):
    """
    Executa uma estrutura com e sem amortecimento.
    Calibra Rayleigh com ζ₁=ζ₂=2% nos modos 1 e 2 (típico aço estrutural).
    """
    print('\n' + '=' * 110)
    print(f'  {name}   (ζ₁ = ζ₂ = {short(zeta1*100, 4)}%)')
    print('=' * 110)

    with Timer() as t_a:
        K0, M1, M2 = struct.assemble_K0_M1_M2()
    with Timer() as t_u:
        omegas_und, _ = solve_1mm(K0, M1)

    n_modes = min(5, len(omegas_und))
    if n_modes < 2:
        print('  (Estrutura com menos de 2 modos — pular amortecimento)')
        return

    # Para Rayleigh precisamos 2 frequências distintas (det≠0)
    w_a = omegas_und[0]
    w_b = None
    for w in omegas_und[1:]:
        if abs(w - w_a) > w_a * mpf('1e-6'):
            w_b = w; break
    if w_b is None:
        print('  (Sem 2 frequências distintas — Rayleigh impossível)')
        return
    with Timer() as t_c:
        C, alpha, beta = rayleigh_damping(M1, K0, w_a, w_b, zeta1, zeta2)

    with Timer() as t_d:
        modes_damped = damped_modal_frequencies(M1, C, K0)

    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · und. 1MM={float(t_u.elapsed):.3f}s '
          f'· Rayleigh={float(t_c.elapsed):.3f}s · damped={float(t_d.elapsed):.3f}s')
    print(f'  Rayleigh: α = {short(alpha, 8)} (s⁻¹) · β = {short(beta, 8)} (s)')
    print()
    print(f'  {"Modo":>4} | {"ω_n (não-amort)":>20} | {"ω_n (amort)":>20} | {"ω_d":>20} | {"ζ":>15} | {"Δω/ω_n":>15}')
    print('  ' + '-' * 110)
    for i in range(min(n_modes, len(modes_damped))):
        wn_u = omegas_und[i]
        wn_d, wd, z = modes_damped[i]
        drel = (wn_u - wn_d) / wn_u if wn_u > 0 else mpf(0)
        print(f'  {i+1:>4} | {short(wn_u, 12):>20} | {short(wn_d, 12):>20} | '
              f'{short(wd, 12):>20} | {short(z*100, 8):>14}% | {short(drel*100, 6):>14}%')


def run_all():
    t_g = Timer()
    with t_g:
        # Viga em balanço primeiro (validação)
        s, *_ = cantilever_beam(n_elem=4)  # 4 elem para ficar rápido
        run_example('VIGA EM BALANÇO (validação)', s)

        # Ex 01
        s = ex01_truss_weaver_TQ(nMM=2)
        run_example('EXEMPLO 01 - Treliça Weaver TQ', s)

        # Ex 04
        s = ex04_frame_paz_1no()
        run_example('EXEMPLO 04 - Pórtico 3D Paz', s)

        # Ex 06
        s = ex06_truss3d_paz_1no()
        run_example('EXEMPLO 06 - Treliça 3D Paz', s)

    print('\n' + '=' * 110)
    print(f'TEMPO TOTAL (amortecimento): {float(t_g.elapsed):.2f} s')
    print('=' * 110)


if __name__ == '__main__':
    run_all()
