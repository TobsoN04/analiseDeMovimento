"""
hp.validation — Comparação completa com tese e literatura.

Reproduz todas as tabelas 5.x da dissertação com:
- 50 dígitos de precisão (mpmath)
- Tempo de execução por etapa
- Comparação frequência (ω_atual vs ω_tese vs ω_lit)
- Δω absoluta e relativa
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from mpmath import mp, mpf, sqrt
from hp.core import solve_1mm, solve_2mm, Timer
from hp.examples import (
    ex01_truss_weaver_1mm, ex01_truss_weaver_TQ,
    ex03_frame_weaver_4nos,
    ex04_frame_paz_1no,
    ex06_truss3d_paz_1no,
    cantilever_beam, cantilever_analytical_freqs,
    fmt, short,
)


# Valores extraídos das tabelas da tese (rad/s salvo indicação)
THESIS_DATA = {
    # Tabela 5.1: Treliça Weaver, Weaver vs 1MM
    'ex01_5_1': {
        'unit': 'rad/s',
        'weaver': [mpf('419.95'), mpf('1167.70'), mpf('1861.80')],
        '1MM':    [mpf('420.51'), mpf('1168.20'), mpf('1864.34')],
    },
    # Tabela 5.3: Treliça Weaver com perfis TQ
    'ex01_5_3': {
        'unit': 'rad/s',
        '1MM': [mpf('420.51'), mpf('1168.20'), mpf('1864.34')],
        '2MM': [mpf('297.28'), mpf('950.53'),  mpf('1078.20')],
        '3MM': [mpf('253.72'), mpf('666.71'),  mpf('985.98')],
        '4MM': [mpf('233.10'), mpf('544.95'),  mpf('934.69')],
        '5MM': [mpf('221.49'), mpf('482.21'),  mpf('899.44')],
        '6MM': [mpf('214.23'), mpf('444.73'),  mpf('874.57')],
    },
    # Tabela 5.5: Pórtico Weaver, em Hz
    'ex03_5_5': {
        'unit': 'Hz',
        'weaver': [mpf('79.55'), mpf('168.90')],
        '1MM':    [mpf('79.55'), mpf('168.88')],
    },
    # Tabela 5.9: Pórtico Weaver, em rad/s
    'ex03_5_9': {
        'unit': 'rad/s',
        '1MM': [mpf('89.39'),  mpf('182.46'), mpf('374.18')],
        '2MM': [mpf('89.12'),  mpf('181.45'), mpf('318.79')],
        '3MM': [mpf('89.11'),  mpf('181.39'), mpf('304.74')],
        '4MM': [mpf('89.11'),  mpf('181.38'), mpf('299.57')],
        '5MM': [mpf('89.11'),  mpf('181.38'), mpf('297.37')],
        '6MM': [mpf('88.86'),  mpf('180.87'), mpf('295.55')],
    },
    # Tabela 5.12: Pórtico 3D Paz
    'ex04_5_12': {
        'unit': 'rad/s',
        'paz': [mpf('80.50'), mpf('80.70'), mpf('88.60'),
                mpf('417.81'), mpf('489.36'), mpf('517.15')],
        '1MM': [mpf('80.54'), mpf('80.70'), mpf('88.64'),
                mpf('417.81'), mpf('489.47'), mpf('517.23')],
        '2MM': [mpf('65.80'), mpf('65.87'), mpf('72.87'),
                mpf('226.89'), mpf('257.83'), mpf('279.60')],
        '3MM': [mpf('62.06'), mpf('62.10'), mpf('69.07'),
                mpf('162.43'), mpf('174.46'), mpf('192.88')],
        '4MM': [mpf('60.65'), mpf('60.68'), mpf('67.74'),
                mpf('134.95'), mpf('140.80'), mpf('157.79')],
        '5MM': [mpf('60.01'), mpf('60.05'), mpf('67.21'),
                mpf('120.21'), mpf('123.47'), mpf('139.86')],
        '6MM': [mpf('59.70'), mpf('59.73'), mpf('66.97'),
                mpf('111.10'), mpf('113.09'), mpf('129.27')],
    },
    # Tabela 5.16: Treliça 3D Paz (em Hz)
    'ex06_5_16': {
        'unit': 'Hz',
        'paz': [mpf('32.84'), mpf('69.15'), mpf('98.95')],
        '1MM': [mpf('32.84'), mpf('69.15'), mpf('98.95')],
    },
    # Tabela 5.18: Treliça 3D Paz com perfil TQ (Hz)
    'ex06_5_18': {
        'unit': 'Hz',
        '1MM': [mpf('32.84'), mpf('69.15'), mpf('98.95')],
        '2MM': [mpf('28.58'), mpf('48.94'), mpf('77.07')],
        '3MM': [mpf('26.99'), mpf('40.86'), mpf('62.23')],
        '4MM': [mpf('26.22'), mpf('36.93'), mpf('53.15')],
        '5MM': [mpf('25.79'), mpf('34.68'), mpf('47.66')],
        '6MM': [mpf('25.54'), mpf('33.26'), mpf('44.15')],
    },
}


# =============================================================================
# Comparador
# =============================================================================

def diff(a, b):
    """Diferença absoluta e relativa em %."""
    da = abs(a - b)
    if b != 0:
        dr = da / abs(b) * 100
    else:
        dr = mpf('inf')
    return da, dr


def print_comparison_row(modo, atual, ref_dict, n_dig=8):
    parts = [f'{modo:>4}']
    parts.append(f'{short(atual, n_dig):>15}')
    for label, ref in ref_dict.items():
        da, dr = diff(atual, ref)
        parts.append(f'{short(ref, n_dig):>12}')
        parts.append(f'{short(dr, 5):>10}%')
    return ' | '.join(parts)


# =============================================================================
# Validações
# =============================================================================

def validate_cantilever():
    """Viga em balanço vs solução analítica de Euler-Bernoulli."""
    print('\n' + '=' * 110)
    print('VALIDAÇÃO 0: VIGA EM BALANÇO — solução analítica de Euler-Bernoulli')
    print('=' * 110)
    s, E, I, rho, A, L = cantilever_beam(n_elem=10)

    t_total = Timer()
    with t_total:
        with Timer() as t_a:
            K0, M1, M2 = s.assemble_K0_M1_M2()
        with Timer() as t_1:
            omegas_1, _ = solve_1mm(K0, M1)
        with Timer() as t_2:
            omegas_2, _ = solve_2mm(K0, M1, M2)
    f_anal = cantilever_analytical_freqs()

    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · 1MM={float(t_1.elapsed):.3f}s '
          f'· 2MM={float(t_2.elapsed):.3f}s · total={float(t_total.elapsed):.3f}s')
    print()
    print(f'  {"Modo":>4} | {"Analítico (35 dig)":>40} | {"1MM (35 dig)":>40} | {"Erro 1MM":>10} | {"2MM (35 dig)":>40} | {"Erro 2MM":>10}')
    print('  ' + '-' * 156)
    for i in range(min(3, len(omegas_1), len(omegas_2))):
        da1, dr1 = diff(omegas_1[i], f_anal[i])
        da2, dr2 = diff(omegas_2[i], f_anal[i])
        print(f'  {i+1:>4} | {fmt(f_anal[i], 35):>40} | {fmt(omegas_1[i], 35):>40} | {short(dr1, 5):>9}% | {fmt(omegas_2[i], 35):>40} | {short(dr2, 5):>9}%')


def validate_ex01():
    print('\n' + '=' * 110)
    print('EXEMPLO 01 — TRELIÇA PLANA WEAVER (Tabelas 5.1 e 5.3)')
    print('=' * 110)

    # ----- Tabela 5.1: 1MM com A único -----
    print('\n[Tabela 5.1] Geometria original Weaver (A=6.451e-3 única)')
    s = ex01_truss_weaver_1mm()
    with Timer() as t_a:
        K0, M1, M2 = s.assemble_K0_M1_M2()
    with Timer() as t_1:
        omegas, _ = solve_1mm(K0, M1)
    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · 1MM={float(t_1.elapsed):.3f}s')

    ref = THESIS_DATA['ex01_5_1']
    print(f'  {"Modo":>4} | {"Atual (35 dig)":>40} | {"Tese 1MM":>10} | {"Δ% tese":>9} | {"Weaver":>10} | {"Δ% Weav":>9}')
    print('  ' + '-' * 95)
    for i in range(min(3, len(omegas))):
        da1, dr1 = diff(omegas[i], ref['1MM'][i])
        da2, dr2 = diff(omegas[i], ref['weaver'][i])
        print(f'  {i+1:>4} | {fmt(omegas[i], 35):>40} | {short(ref["1MM"][i],8):>10} | {short(dr1,4):>8}% | {short(ref["weaver"][i],8):>10} | {short(dr2,4):>8}%')

    # ----- Tabela 5.3: 1MM e 2MM com perfis TQ -----
    print('\n[Tabela 5.3] Perfis TQ diferentes por barra (Tabela 5.2)')
    s = ex01_truss_weaver_TQ(nMM=2)
    with Timer() as t_a:
        K0, M1, M2 = s.assemble_K0_M1_M2()
    with Timer() as t_1:
        omegas_1, _ = solve_1mm(K0, M1)
    with Timer() as t_2:
        omegas_2, _ = solve_2mm(K0, M1, M2)
    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · 1MM={float(t_1.elapsed):.3f}s · 2MM={float(t_2.elapsed):.3f}s')

    ref = THESIS_DATA['ex01_5_3']
    print(f'  {"Modo":>4} | {"1MM atual":>15} | {"Tese 1MM":>10} | {"Δ% 1MM":>9} | {"2MM atual":>15} | {"Tese 2MM":>10} | {"Δ% 2MM":>9}')
    print('  ' + '-' * 110)
    for i in range(min(3, len(omegas_1))):
        da1, dr1 = diff(omegas_1[i], ref['1MM'][i])
        if i < len(omegas_2):
            da2, dr2 = diff(omegas_2[i], ref['2MM'][i])
            o2_str = short(omegas_2[i], 8); dr2_str = short(dr2, 4)
        else:
            o2_str = '—'; dr2_str = '—'
        print(f'  {i+1:>4} | {short(omegas_1[i],10):>15} | {short(ref["1MM"][i],8):>10} | {short(dr1,4):>8}% | {o2_str:>15} | {short(ref["2MM"][i],8):>10} | {dr2_str:>8}%')


def validate_ex03():
    print('\n' + '=' * 110)
    print('EXEMPLO 03 — PÓRTICO 2D WEAVER (Tabelas 5.5 e 5.9)')
    print('=' * 110)
    s = ex03_frame_weaver_4nos()
    with Timer() as t_a:
        K0, M1, M2 = s.assemble_K0_M1_M2()
    with Timer() as t_1:
        omegas_1, _ = solve_1mm(K0, M1)
    with Timer() as t_2:
        omegas_2, _ = solve_2mm(K0, M1, M2)
    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · 1MM={float(t_1.elapsed):.3f}s · 2MM={float(t_2.elapsed):.3f}s')

    ref = THESIS_DATA['ex03_5_9']
    print(f'  {"Modo":>4} | {"1MM atual":>15} | {"Tese 1MM":>10} | {"Δ% 1MM":>9} | {"2MM atual":>15} | {"Tese 2MM":>10} | {"Δ% 2MM":>9}')
    print('  ' + '-' * 110)
    for i in range(min(3, len(omegas_1))):
        da1, dr1 = diff(omegas_1[i], ref['1MM'][i])
        if i < len(omegas_2):
            da2, dr2 = diff(omegas_2[i], ref['2MM'][i])
            o2_str = short(omegas_2[i], 8); dr2_str = short(dr2, 4)
        else:
            o2_str = '—'; dr2_str = '—'
        print(f'  {i+1:>4} | {short(omegas_1[i],10):>15} | {short(ref["1MM"][i],8):>10} | {short(dr1,4):>8}% | {o2_str:>15} | {short(ref["2MM"][i],8):>10} | {dr2_str:>8}%')


def validate_ex04():
    print('\n' + '=' * 110)
    print('EXEMPLO 04 — PÓRTICO 3D PAZ (Tabela 5.12)')
    print('=' * 110)
    s = ex04_frame_paz_1no()
    with Timer() as t_a:
        K0, M1, M2 = s.assemble_K0_M1_M2()
    with Timer() as t_1:
        omegas_1, _ = solve_1mm(K0, M1)
    with Timer() as t_2:
        omegas_2, _ = solve_2mm(K0, M1, M2)
    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · 1MM={float(t_1.elapsed):.3f}s · 2MM={float(t_2.elapsed):.3f}s')

    ref = THESIS_DATA['ex04_5_12']
    print(f'  {"Modo":>4} | {"1MM atual":>15} | {"Tese 1MM":>10} | {"Δ% 1MM":>9} | {"Paz":>8} | {"Δ% Paz":>9}')
    print('  ' + '-' * 95)
    for i in range(min(6, len(omegas_1))):
        da1, dr1 = diff(omegas_1[i], ref['1MM'][i])
        da2, dr2 = diff(omegas_1[i], ref['paz'][i])
        print(f'  {i+1:>4} | {short(omegas_1[i],10):>15} | {short(ref["1MM"][i],8):>10} | {short(dr1,4):>8}% | {short(ref["paz"][i],8):>8} | {short(dr2,4):>8}%')

    print()
    print(f'  {"Modo":>4} | {"2MM atual":>15} | {"Tese 2MM":>10} | {"Δ% 2MM":>9} | {"Tese 6MM":>10}')
    print('  ' + '-' * 75)
    for i in range(min(6, len(omegas_2))):
        da, dr = diff(omegas_2[i], ref['2MM'][i])
        print(f'  {i+1:>4} | {short(omegas_2[i],10):>15} | {short(ref["2MM"][i],8):>10} | {short(dr,4):>8}% | {short(ref["6MM"][i],8):>10}')


def validate_ex06():
    print('\n' + '=' * 110)
    print('EXEMPLO 06 — TRELIÇA 3D PAZ (Tabela 5.16)')
    print('=' * 110)
    s = ex06_truss3d_paz_1no()
    with Timer() as t_a:
        K0, M1, M2 = s.assemble_K0_M1_M2()
    with Timer() as t_1:
        omegas_1, _ = solve_1mm(K0, M1)
    with Timer() as t_2:
        omegas_2, _ = solve_2mm(K0, M1, M2)
    print(f'  Tempos: assembly={float(t_a.elapsed):.3f}s · 1MM={float(t_1.elapsed):.3f}s · 2MM={float(t_2.elapsed):.3f}s')

    # Converter rad/s para Hz
    two_pi = 2 * mp.pi
    omegas_1_Hz = [w / two_pi for w in omegas_1]
    omegas_2_Hz = [w / two_pi for w in omegas_2]

    ref = THESIS_DATA['ex06_5_16']
    print(f'  {"Modo":>4} | {"1MM (Hz) atual":>20} | {"Tese/Paz":>10} | {"Δ%":>9}')
    print('  ' + '-' * 60)
    for i in range(min(3, len(omegas_1_Hz))):
        da, dr = diff(omegas_1_Hz[i], ref['paz'][i])
        print(f'  {i+1:>4} | {short(omegas_1_Hz[i],10):>20} | {short(ref["paz"][i],8):>10} | {short(dr,4):>8}%')


# =============================================================================
# Runner
# =============================================================================

def run_all():
    t_global = Timer()
    with t_global:
        validate_cantilever()
        validate_ex01()
        validate_ex03()
        validate_ex04()
        validate_ex06()
    print('\n' + '=' * 110)
    print(f'TEMPO TOTAL DE EXECUÇÃO: {float(t_global.elapsed):.2f} s')
    print('=' * 110)


if __name__ == '__main__':
    run_all()
