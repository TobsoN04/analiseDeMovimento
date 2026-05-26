"""
hp.full_validation — Reprodução completa de todas as tabelas da tese.

Para cada exemplo (todas as discretizações):
- Roda 1MM, 2MM, 3MM, 4MM em alta precisão
- Tempo de execução por nMM
- Comparação com tese (Tabelas 5.x) e literatura (Weaver/Paz/Petyt)
- Δ frequência e erro %
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from mpmath import mp, mpf, sqrt
mp.dps = 60  # aumentar precisão para nMM altos

from hp.core import Timer
from hp.nmm_solver import solve_struct_nmm
from hp.examples_disc import (
    ex01_2nos, ex01_subdivided,
    ex03_4nos,
    ex04_1no, ex04_13nos,
    ex06_1no, ex06_15nos,
)
from hp.examples import cantilever_beam, cantilever_analytical_freqs, short


# Valores extraídos das tabelas da tese
THESIS = {
    '5.1': {  # Treliça Weaver 2 nós - 1MM vs Weaver
        'unit': 'rad/s',
        'weaver': [420, 1167.70, 1861.80],  # 419.95
        '1MM':    [mpf('420.51'), mpf('1168.20'), mpf('1864.34')],
    },
    '5.3': {  # Treliça Weaver TQ
        'unit': 'rad/s',
        '1MM': [mpf('420.51'), mpf('1168.20'), mpf('1864.34')],
        '2MM': [mpf('297.28'), mpf('950.53'),  mpf('1078.20')],
        '3MM': [mpf('253.72'), mpf('666.71'),  mpf('985.98')],
        '4MM': [mpf('233.10'), mpf('544.95'),  mpf('934.69')],
        '5MM': [mpf('221.49'), mpf('482.21'),  mpf('899.44')],
        '6MM': [mpf('214.23'), mpf('444.73'),  mpf('874.57')],
    },
    '5.4': {  # Treliça 5 nós (Weaver discretizada)
        'unit': 'rad/s',
        '1MM': [mpf('315.82'), mpf('587.47'), mpf('806.24')],
        '2MM': [mpf('313.27'), mpf('579.91'), mpf('797.63')],
        '3MM': [mpf('313.20'), mpf('579.48'), mpf('797.09')],
        '4MM': [mpf('313.20'), mpf('579.45'), mpf('797.04')],
        '5MM': [mpf('313.20'), mpf('579.45'), mpf('797.03')],
        '6MM': [mpf('313.20'), mpf('579.45'), mpf('797.03')],
    },
    '5.9': {  # Pórtico Weaver 4 nós
        'unit': 'rad/s',
        '1MM': [mpf('89.39'), mpf('182.46'), mpf('374.18')],
        '2MM': [mpf('89.12'), mpf('181.45'), mpf('318.79')],
        '3MM': [mpf('89.11'), mpf('181.39'), mpf('304.74')],
        '4MM': [mpf('89.11'), mpf('181.38'), mpf('299.57')],
        '5MM': [mpf('89.11'), mpf('181.38'), mpf('297.37')],
        '6MM': [mpf('88.86'), mpf('180.87'), mpf('295.55')],
    },
    '5.10': {  # Pórtico Weaver 10 nós
        'unit': 'rad/s',
        '1MM': [mpf('87.39'), mpf('173.33'), mpf('245.32')],
        '6MM': [mpf('87.30'), mpf('173.27'), mpf('284.87')],
    },
    '5.12': {  # Pórtico 3D Paz 1 nó
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
    '5.13': {  # Pórtico 3D Paz 13 nós
        'unit': 'rad/s',
        '1MM': [mpf('56.46'), mpf('58.88'), mpf('64.77')],
        '6MM': [mpf('56.43'), mpf('58.85'), mpf('64.72')],
    },
    '5.16': {  # Treliça 3D Paz 1 nó (Hz)
        'unit': 'Hz',
        'paz': [mpf('32.84'), mpf('69.15'), mpf('98.95')],
        '1MM': [mpf('32.84'), mpf('69.15'), mpf('98.95')],
    },
    '5.18': {  # Treliça 3D Paz TQ
        'unit': 'Hz',
        '1MM': [mpf('32.84'), mpf('69.15'), mpf('98.95')],
        '2MM': [mpf('28.58'), mpf('48.94'), mpf('77.07')],
        '3MM': [mpf('26.99'), mpf('40.86'), mpf('62.23')],
        '4MM': [mpf('26.22'), mpf('36.93'), mpf('53.15')],
        '5MM': [mpf('25.79'), mpf('34.68'), mpf('47.66')],
        '6MM': [mpf('25.54'), mpf('33.26'), mpf('44.15')],
    },
    '5.20': {  # Treliça 3D Paz 15 nós TQ
        'unit': 'Hz',
        '1MM': [mpf('33.39'), mpf('39.79'), mpf('48.28')],
        '6MM': [mpf('33.35'), mpf('39.69'), mpf('48.12')],
    },
}


def run_struct(struct, name, nMM_max=4, ref_data=None, unit_factor=mpf(1)):
    """Roda struct para nMM=1..nMM_max e compara com ref_data."""
    print(f'\n{"=" * 110}')
    print(f'  {name}')
    print('=' * 110)

    results_per_mm = {}
    tempos = {}
    for n in range(1, nMM_max + 1):
        try:
            with Timer() as t:
                omegas, _ = solve_struct_nmm(struct, n)
            results_per_mm[n] = omegas[:6] if len(omegas) >= 6 else omegas
            tempos[n] = float(t.elapsed)
        except Exception as ex:
            results_per_mm[n] = []
            tempos[n] = -1
            print(f'  Erro em {n}MM: {ex}')

    # tabela de frequências (em unit_factor já aplicado)
    # ex.: para Ex 06 unit_factor = 1/(2π) para converter rad/s → Hz
    print(f'\n  {"Modo":>4}' + ''.join(f' | {f"{n}MM atual":>18}' for n in range(1, nMM_max + 1)))
    print('  ' + '-' * (4 + nMM_max * 21))
    n_show = max((len(v) for v in results_per_mm.values()), default=0)
    n_show = min(n_show, 6)
    for i in range(n_show):
        cols = [f'{i+1:>4}']
        for n in range(1, nMM_max + 1):
            v = results_per_mm.get(n, [])
            if i < len(v):
                cols.append(f'{short(v[i]*unit_factor, 10):>18}')
            else:
                cols.append(f'{"—":>18}')
        print('  ' + ' | '.join(cols))
    print(f'\n  Tempos por nMM (s):')
    for n in range(1, nMM_max + 1):
        print(f'    {n}MM: {tempos[n]:.3f}s' + ('' if tempos[n] > 0 else ' (FALHOU)'))

    # comparação com tese
    if ref_data is not None:
        print(f'\n  Comparação com tese:')
        print(f'  {"Modo":>4}' + ''.join(f' | {f"{n}MM Δ%":>12}' for n in range(1, nMM_max + 1)))
        print('  ' + '-' * (4 + nMM_max * 15))
        for i in range(min(6, n_show)):
            cols = [f'{i+1:>4}']
            for n in range(1, nMM_max + 1):
                v = results_per_mm.get(n, [])
                tese = ref_data.get(f'{n}MM')
                if i < len(v) and tese is not None and i < len(tese):
                    err = abs(v[i] * unit_factor - tese[i]) / tese[i] * 100
                    cols.append(f'{short(err, 6):>11}%')
                else:
                    cols.append(f'{"—":>12}')
            print('  ' + ' | '.join(cols))


def run_cantilever():
    """Validação contra solução analítica de Euler-Bernoulli."""
    print('\n' + '=' * 110)
    print('  VIGA EM BALANÇO — validação contra solução analítica')
    print('=' * 110)
    f_anal = cantilever_analytical_freqs()
    print(f'  Analítico (rad/s): {[short(f, 10) for f in f_anal[:3]]}')

    s, *_ = cantilever_beam(n_elem=4)
    for n in [1, 2, 3, 4]:
        with Timer() as t:
            omegas, _ = solve_struct_nmm(s, n)
        errs = [abs(omegas[i] - f_anal[i]) / f_anal[i] * 100 for i in range(3)]
        print(f'  {n}MM (t={float(t.elapsed):.2f}s) → '
              f'{short(omegas[0], 14)} (Δ {short(errs[0], 5)}%) · '
              f'{short(omegas[1], 14)} (Δ {short(errs[1], 5)}%) · '
              f'{short(omegas[2], 14)} (Δ {short(errs[2], 5)}%)')


def main(nMM_max=4):
    t_g = Timer()
    with t_g:
        run_cantilever()

        # === Ex 01 — Treliça Weaver ===
        run_struct(ex01_2nos(),
                   'EX 01 — Treliça Weaver, 2 nós livres (Tabela 5.3)',
                   nMM_max=nMM_max, ref_data=THESIS['5.3'])

        # === Ex 01 discretizada (5 nós aprox) ===
        run_struct(ex01_subdivided(subdiv=2),
                   'EX 01 — Treliça Weaver discretizada (Tabela 5.4)',
                   nMM_max=nMM_max, ref_data=THESIS['5.4'])

        # === Ex 03 — Pórtico Weaver 4 nós ===
        run_struct(ex03_4nos(),
                   'EX 03 — Pórtico 2D Weaver, 4 nós (Tabela 5.9)',
                   nMM_max=nMM_max, ref_data=THESIS['5.9'])

        # === Ex 04 — Pórtico 3D Paz 1 nó ===
        run_struct(ex04_1no(),
                   'EX 04 — Pórtico 3D Paz, 1 nó (Tabela 5.12)',
                   nMM_max=nMM_max, ref_data=THESIS['5.12'])

        # === Ex 06 — Treliça 3D Paz 1 nó (Hz) ===
        from mpmath import pi as mp_pi
        run_struct(ex06_1no(),
                   'EX 06 — Treliça 3D Paz, 1 nó (Tabela 5.18, Hz)',
                   nMM_max=nMM_max, ref_data=THESIS['5.18'],
                   unit_factor=1/(2 * mp_pi))

    print(f'\n{"=" * 110}')
    print(f'  TEMPO TOTAL: {float(t_g.elapsed):.2f} s · precisão {mp.dps} dígitos')
    print('=' * 110)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nmm', type=int, default=6, help='Máximo número de matrizes de massa (1-6)')
    args = p.parse_args()
    main(nMM_max=args.nmm)
