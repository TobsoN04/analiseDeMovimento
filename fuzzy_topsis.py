"""
fuzzy_topsis — Análise Fuzzy-TOPSIS para classificar cada exemplo:
   ALT1 = "Precisa de mais NÓS (discretização)"
   ALT2 = "Precisa de mais MATRIZES DE MASSA (nMM)"
   ALT3 = "Convergiu (não precisa de mais)"

Critérios usados (todos extraídos automaticamente da execução):
   C1: gradiente_freq vs nMM   — quanto a freq muda de 1MM → 2MM (benefício: alto → precisa mais MM)
   C2: gradiente_freq vs nós   — quanto a freq mudaria com mais discretização (proxy: razão modos finais/iniciais)
   C3: erro_relativo_vs_literatura — diferença % vs valores Weaver/Paz (custo: alto = ruim)
   C4: tempo_execucao          — custo de processamento (custo: alto = ruim)
   C5: amortecimento_efetivo   — razão de amortecimento ζ (informativo)

Procedimento Fuzzy-TOPSIS clássico (Chen, 2000):
   1. Construir matriz de decisão fuzzy (triangular: L, M, U)
   2. Normalizar
   3. Aplicar pesos (também fuzzy)
   4. Calcular distâncias até a solução fuzzy positiva ideal (A+) e negativa (A-)
   5. Coeficiente de proximidade CC = d-/(d+ + d-)
   6. Ranking pela maior CC

Implementado em alta precisão via mpmath.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from mpmath import mp, mpf, sqrt
mp.dps = 50

from hp.core import Timer, solve_1mm, solve_2mm
from hp.examples import (
    ex01_truss_weaver_TQ, ex03_frame_weaver_4nos,
    ex04_frame_paz_1no, ex06_truss3d_paz_1no, cantilever_beam,
    short,
)
from hp.examples_refined import (
    ex01_truss_weaver_TQ_refined, ex04_frame_paz_refined,
    ex06_truss3d_paz_refined,
)
from damped.core import rayleigh_damping, damped_modal_frequencies


# =============================================================================
# Números fuzzy triangulares (L, M, U)
# =============================================================================

class TFN:
    """Triangular Fuzzy Number (L, M, U)."""
    __slots__ = ('L', 'M', 'U')
    def __init__(self, L, M, U):
        self.L = mpf(L); self.M = mpf(M); self.U = mpf(U)
    def __repr__(self):
        return f'TFN({short(self.L,4)}, {short(self.M,4)}, {short(self.U,4)})'
    def __mul__(self, other):
        if isinstance(other, TFN):
            return TFN(self.L*other.L, self.M*other.M, self.U*other.U)
        c = mpf(other)
        return TFN(self.L*c, self.M*c, self.U*c)
    __rmul__ = __mul__
    def __truediv__(self, c):
        c = mpf(c)
        return TFN(self.L/c, self.M/c, self.U/c)


def tfn_distance(a, b):
    """Distância vertex entre duas TFNs (d_v)."""
    return sqrt(((a.L - b.L)**2 + (a.M - b.M)**2 + (a.U - b.U)**2) / 3)


# Variáveis linguísticas → TFNs (escala 0-1)
LING_TO_TFN = {
    'MB': TFN('0.0', '0.1', '0.2'),   # Muito Baixo
    'B':  TFN('0.1', '0.3', '0.5'),   # Baixo
    'M':  TFN('0.3', '0.5', '0.7'),   # Médio
    'A':  TFN('0.5', '0.7', '0.9'),   # Alto
    'MA': TFN('0.7', '0.9', '1.0'),   # Muito Alto
}


# Pesos dos critérios (linguísticos)
WEIGHTS = {
    'C1_grad_MM':  LING_TO_TFN['MA'],   # gradiente com MM — peso alto
    'C2_grad_nos': LING_TO_TFN['MA'],   # gradiente com nós — peso alto
    'C3_erro_lit': LING_TO_TFN['A'],    # erro vs literatura — peso alto
    'C4_tempo':    LING_TO_TFN['B'],    # tempo de execução — peso baixo
    'C5_amort':    LING_TO_TFN['M'],    # amortecimento — peso médio
}

CRITERIA_KEYS = list(WEIGHTS.keys())
# tipo de critério: True = benefício (maior é melhor), False = custo
CRITERIA_TYPE = [True, True, False, False, True]


# =============================================================================
# Coleta de dados de cada exemplo
# =============================================================================

def value_to_tfn(v, vmin, vmax):
    """Converte valor numérico v em TFN normalizado ao intervalo [vmin, vmax]."""
    if vmax <= vmin:
        return LING_TO_TFN['M']
    x = (mpf(v) - vmin) / (vmax - vmin)
    # criar TFN centrado em x com largura ±0.1
    L = max(mpf(0), x - mpf('0.1'))
    U = min(mpf(1), x + mpf('0.1'))
    M = x
    return TFN(L, M, U)


def analyze_example(name, struct, ref_lit_freqs=None, struct_refined=None):
    """Roda 1MM e 2MM, mede tempo, calcula critérios numéricos.

    Se struct_refined for fornecido, mede também a variação da freq fundamental
    entre as duas discretizações → critério C2.
    """
    print(f'\n[Análise] {name}')
    with Timer() as t_a:
        K0, M1, M2 = struct.assemble_K0_M1_M2()
    with Timer() as t_1:
        omegas_1, _ = solve_1mm(K0, M1)
    with Timer() as t_2:
        omegas_2, _ = solve_2mm(K0, M1, M2)
    total_time = float(t_a.elapsed + t_1.elapsed + t_2.elapsed)

    # C1: gradiente entre 1MM e 2MM (médio relativo dos primeiros modos)
    n_cmp = min(3, len(omegas_1), len(omegas_2))
    grad_MM = sum(abs(omegas_1[i] - omegas_2[i]) / omegas_1[i]
                   for i in range(n_cmp)) / n_cmp

    # C2: gradiente com discretização — usar struct refinada se fornecida
    if struct_refined is not None:
        with Timer() as t_r:
            K0r, M1r, _ = struct_refined.assemble_K0_M1_M2()
            omegas_r, _ = solve_1mm(K0r, M1r)
        n_cmp_r = min(3, len(omegas_1), len(omegas_r))
        grad_nos = sum(abs(omegas_1[i] - omegas_r[i]) / omegas_1[i]
                        for i in range(n_cmp_r)) / n_cmp_r
        total_time += float(t_r.elapsed)
    else:
        grad_nos = mpf(0)

    # C3: erro relativo vs literatura (se fornecido)
    if ref_lit_freqs:
        n_cmp = min(len(ref_lit_freqs), len(omegas_1))
        erro = sum(abs(omegas_1[i] - ref_lit_freqs[i]) / ref_lit_freqs[i]
                   for i in range(n_cmp)) / n_cmp
    else:
        erro = mpf('0.05')  # default 5% se não há referência

    # C4: tempo de execução normalizado
    tempo = mpf(total_time)

    # C5: amortecimento efetivo (calcular ζ com Rayleigh 2%)
    if len(omegas_1) >= 2:
        w_a = omegas_1[0]
        w_b = None
        for w in omegas_1[1:]:
            if abs(w - w_a) > w_a * mpf('1e-6'):
                w_b = w; break
        if w_b:
            C, _, _ = rayleigh_damping(M1, K0, w_a, w_b, mpf('0.02'), mpf('0.02'))
            modes_d = damped_modal_frequencies(M1, C, K0)
            zetas = [z for (_, _, z) in modes_d[:n_cmp]]
            amort = sum(zetas, mpf(0)) / max(1, len(zetas))
        else:
            amort = mpf('0.02')
    else:
        amort = mpf('0.02')

    return {
        'name': name,
        'C1_grad_MM': grad_MM,
        'C2_grad_nos': grad_nos,
        'C3_erro_lit': erro,
        'C4_tempo': tempo,
        'C5_amort': amort,
        'omegas_1': omegas_1,
        'omegas_2': omegas_2,
        'time_total': total_time,
    }


# =============================================================================
# Fuzzy-TOPSIS — algoritmo completo
# =============================================================================

def build_fuzzy_decision_matrix(data_rows):
    """
    Para cada critério, encontra o mín/máx entre as estruturas analisadas,
    e converte cada valor em TFN normalizado.
    """
    matrix_F = []  # matrix_F[i] = lista de TFN para estrutura i
    # min/max por critério
    bounds = {}
    for key in CRITERIA_KEYS:
        vals = [d[key] for d in data_rows]
        bounds[key] = (min(vals), max(vals))
    for row in data_rows:
        tfns = [value_to_tfn(row[k], *bounds[k]) for k in CRITERIA_KEYS]
        matrix_F.append(tfns)
    return matrix_F, bounds


def normalize_fuzzy(F, types):
    """Normalização linear de TFNs (Chen 2000)."""
    n_alt = len(F)
    n_crit = len(F[0])
    R = [[None]*n_crit for _ in range(n_alt)]
    for j in range(n_crit):
        col = [F[i][j] for i in range(n_alt)]
        if types[j]:  # benefício
            cstar = max(c.U for c in col)
            if cstar == 0: cstar = mpf(1)
            for i in range(n_alt):
                R[i][j] = TFN(F[i][j].L / cstar, F[i][j].M / cstar, F[i][j].U / cstar)
        else:  # custo
            cmin = min(c.L for c in col)
            if cmin == 0: cmin = mpf('1e-30')
            for i in range(n_alt):
                F_ij = F[i][j]
                # custo: inverte
                Lv = cmin / F_ij.U if F_ij.U > 0 else mpf(0)
                Mv = cmin / F_ij.M if F_ij.M > 0 else mpf(0)
                Uv = cmin / F_ij.L if F_ij.L > 0 else mpf(1)
                R[i][j] = TFN(Lv, Mv, Uv)
    return R


def weighted_fuzzy(R, W):
    """Aplica pesos fuzzy."""
    n_alt = len(R)
    n_crit = len(R[0])
    V = [[None]*n_crit for _ in range(n_alt)]
    for i in range(n_alt):
        for j in range(n_crit):
            V[i][j] = R[i][j] * W[j]
    return V


def topsis_proximity(V):
    """Coeficiente de proximidade CC_i = d-/(d+ + d-)."""
    # FPIS = (1,1,1), FNIS = (0,0,0) (após pesos normalizados)
    n_alt = len(V)
    n_crit = len(V[0])
    Aplus = TFN(1, 1, 1)
    Aminus = TFN(0, 0, 0)
    d_plus = [mpf(0)] * n_alt
    d_minus = [mpf(0)] * n_alt
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += tfn_distance(V[i][j], Aplus)
            d_minus[i] += tfn_distance(V[i][j], Aminus)
    CC = []
    for i in range(n_alt):
        s = d_plus[i] + d_minus[i]
        CC.append(d_minus[i] / s if s > 0 else mpf(0))
    return CC, d_plus, d_minus


def classify_alternative(CC, row):
    """
    Classifica a estrutura conforme o ranking dos critérios:
    - Se C1 (grad_MM) é alto → precisa de mais MATRIZES DE MASSA
    - Se C2 (grad_nos) é alto → precisa de mais NÓS
    - Se ambos baixos e erro baixo → CONVERGIU
    """
    g_MM = row['C1_grad_MM']
    g_nos = row['C2_grad_nos']
    erro = row['C3_erro_lit']

    THRESH_LO = mpf('0.01')   # 1%

    # ambos gradientes pequenos → convergiu (sem dependência de erro_lit
    # porque pode estar grande por questões de parâmetros/geometria)
    if g_MM < THRESH_LO and g_nos < THRESH_LO:
        return 'CONVERGIU'
    # diferença significativa entre gradientes determina recomendação
    if g_MM > g_nos * mpf('1.1'):
        return 'MAIS_MATRIZES_DE_MASSA'
    if g_nos > g_MM * mpf('1.1'):
        return 'MAIS_NOS'
    return 'AMBOS_RECOMENDADOS'


# =============================================================================
# Pipeline
# =============================================================================

def run_full_analysis():
    print('=' * 110)
    print(' ANÁLISE FUZZY-TOPSIS — Classificação dos exemplos da tese')
    print('=' * 110)

    # Coletar dados de cada exemplo (estrutura base + refinada)
    data = []
    examples = [
        ('Viga em balanço (4 elem)',
         cantilever_beam(n_elem=4)[0],
         cantilever_beam(n_elem=8)[0],
         None),
        ('Ex 01 - Treliça Weaver TQ',
         ex01_truss_weaver_TQ(nMM=2),
         ex01_truss_weaver_TQ_refined(subdiv=2),
         [mpf('419.95'), mpf('1167.70'), mpf('1861.80')]),  # Weaver
        ('Ex 04 - Pórtico 3D Paz',
         ex04_frame_paz_1no(),
         ex04_frame_paz_refined(subdiv=2),
         [mpf('80.50'), mpf('80.70'), mpf('88.60')]),  # Paz
        ('Ex 06 - Treliça 3D Paz',
         ex06_truss3d_paz_1no(),
         ex06_truss3d_paz_refined(subdiv=2),
         [mpf('32.84'), mpf('69.15'), mpf('98.95')]),  # Paz (Hz, converter)
    ]

    t_g = Timer()
    with t_g:
        for name, struct, struct_ref, ref in examples:
            # Para Ex 06 referência está em Hz — converter para rad/s
            if 'Ex 06' in name and ref is not None:
                ref = [r * 2 * mp.pi for r in ref]
            row = analyze_example(name, struct, ref, struct_refined=struct_ref)
            data.append(row)

    # Construir matriz fuzzy
    F, bounds = build_fuzzy_decision_matrix(data)
    R = normalize_fuzzy(F, CRITERIA_TYPE)
    W = [WEIGHTS[k] for k in CRITERIA_KEYS]
    V = weighted_fuzzy(R, W)
    CC, dp, dm = topsis_proximity(V)

    # Reportar
    print(f'\n{"=" * 110}')
    print(' VALORES BRUTOS DOS CRITÉRIOS')
    print('=' * 110)
    print(f'{"Exemplo":<32} | {"C1 grad_MM":>15} | {"C2 grad_nos":>15} | {"C3 erro_lit":>15} | {"C4 tempo(s)":>12} | {"C5 ζ":>10}')
    print('-' * 110)
    for row in data:
        print(f'{row["name"]:<32} | {short(row["C1_grad_MM"]*100, 6):>14}% | '
              f'{short(row["C2_grad_nos"]*100, 6):>14}% | {short(row["C3_erro_lit"]*100, 6):>14}% | '
              f'{short(row["C4_tempo"], 6):>12} | {short(row["C5_amort"]*100, 4):>9}%')

    print(f'\n{"=" * 110}')
    print(' SCORES FUZZY-TOPSIS')
    print('=' * 110)
    print(f'{"Exemplo":<32} | {"d+":>15} | {"d-":>15} | {"CC":>15} | {"Classificação":>30}')
    print('-' * 110)
    # ordenar por CC decrescente
    idx_sorted = sorted(range(len(data)), key=lambda i: -CC[i])
    for rank, i in enumerate(idx_sorted, 1):
        row = data[i]
        cls = classify_alternative(CC[i], row)
        print(f'#{rank} {row["name"]:<29} | {short(dp[i], 8):>15} | {short(dm[i], 8):>15} | '
              f'{short(CC[i], 8):>15} | {cls:>30}')

    print(f'\n{"=" * 110}')
    print(f' Tempo total da análise: {float(t_g.elapsed):.2f} s')
    print('=' * 110)

    print('\n RECOMENDAÇÕES ESPECÍFICAS:')
    for row in data:
        cls = classify_alternative(None, row)
        if cls == 'MAIS_MATRIZES_DE_MASSA':
            recom = f'Aumentar nMM — grad_MM ({short(row["C1_grad_MM"]*100,4)}%) > grad_nós ({short(row["C2_grad_nos"]*100,4)}%)'
        elif cls == 'MAIS_NOS':
            recom = f'Discretizar mais — grad_nós ({short(row["C2_grad_nos"]*100,4)}%) > grad_MM ({short(row["C1_grad_MM"]*100,4)}%)'
        elif cls == 'CONVERGIU':
            recom = 'Convergiu — manter configuração (grad_MM e grad_nós < 1%)'
        elif cls == 'AMBOS_RECOMENDADOS':
            recom = f'Ambos similares — grad_MM={short(row["C1_grad_MM"]*100,4)}%, grad_nós={short(row["C2_grad_nos"]*100,4)}%'
        else:
            recom = 'Inconclusivo'
        print(f'   • {row["name"]:<32}  →  {recom}')


if __name__ == '__main__':
    run_full_analysis()
