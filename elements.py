"""
Matrizes de rigidez efetiva dos elementos unidimensionais.

Baseado no Método Híbrido dos Elementos Finitos (Cap. 4 da dissertação).
Elementos: treliça, viga (Euler-Bernoulli), torção.
Sistemas: 2D e 3D.

Inclui matrizes analíticas K0 (rigidez estática) e M1 (massa consistente clássica),
além das expressões fechadas K(ω) para extração de matrizes de ordem superior.
"""

import numpy as np
from numpy import sin, cos, sinh, cosh, sqrt


# ============================================================================
# Matrizes de rigidez efetiva K(ω) — expressões fechadas
# ============================================================================

def truss_stiffness_1d(E, A, rho, L, omega):
    """Matriz de rigidez efetiva 2x2 do elemento de treliça (Eq. 4-13)."""
    k2 = omega**2 * rho / E
    k = sqrt(abs(k2))
    kL = k * L

    if kL < 0.05:
        return _truss_1d_taylor(E, A, rho, L, omega)

    return (k * E * A / sin(kL)) * np.array([
        [cos(kL), -1],
        [-1, cos(kL)]
    ])


def beam_stiffness_1d(E, I, rho, A, L, omega):
    """
    Matriz de rigidez efetiva 4x4 da viga Euler-Bernoulli.

    Fórmula clássica (Williams & Wittrick, 1983):
        D(ω) = (EI/(γ·L³)) · [aij]
    onde α = βL, β⁴ = ρA·ω²/(EI), γ = 1 - cos(α)·cosh(α).

    Limite ω→0: D(0) = K0 (rigidez estática).
    """
    m = rho * A
    beta4 = omega**2 * m / (E * I)
    beta = abs(beta4)**0.25 if abs(beta4) > 1e-30 else 0.0
    a = beta * L

    if a < 0.05:
        return _beam_1d_taylor(E, I, rho, A, L, omega)

    c = cos(a); C = cosh(a); s = sin(a); S = sinh(a)
    gamma = 1.0 - c * C

    a11 = a**3 * (S*c + C*s)
    a12 = a**2 * L * S * s
    a13 = -a**3 * (s + S)
    a14 = a**2 * L * (C - c)
    a22 = a * L**2 * (C*s - S*c)
    a24 = a * L**2 * (S - s)

    coeff = E * I / (gamma * L**3)
    return coeff * np.array([
        [a11,  a12,  a13,  a14],
        [a12,  a22, -a14,  a24],
        [a13, -a14,  a11, -a12],
        [a14,  a24, -a12,  a22]
    ])


def torsion_stiffness_1d(G, J, rho, Ix, L, omega):
    """Matriz de rigidez efetiva 2x2 do elemento de torção (Eq. 4-48)."""
    Im = rho * Ix
    k2 = omega**2 * Im / (G * J)
    k = sqrt(abs(k2))
    kL = k * L

    if kL < 0.05:
        return _torsion_1d_taylor(G, J, rho, Ix, L, omega)

    return (k * G * J / sin(kL)) * np.array([
        [cos(kL), -1],
        [-1, cos(kL)]
    ])


# ============================================================================
# Expansões de Taylor para kL pequeno (estabilidade numérica)
# ============================================================================

def _truss_1d_taylor(E, A, rho, L, omega):
    """Taylor de K para treliça 1D: K0 - ω²M1 - ω⁴M2 - ω⁶M3."""
    K0, M1 = truss_1d_K0_M1(E, A, rho, L)
    w2 = omega**2
    u2 = w2 * rho * L**2 / E
    K = K0 - w2 * M1
    M2 = (rho * A * L / 6) * (rho * L**2 / (30 * E)) * np.array([
        [8, 7], [7, 8]
    ]) / 8.0
    K = K - w2**2 * M2
    return K


def _beam_1d_taylor(E, I, rho, A, L, omega):
    """Taylor de K para viga 1D: K0 - ω²M1 - ω⁴M2."""
    K0, M1 = beam_1d_K0_M1(E, I, rho, A, L)
    w2 = omega**2
    K = K0 - w2 * M1
    m = rho * A
    L2 = L**2
    M2 = (m**2 * L**5 / (161700 * E * I)) * np.array([
        [59,        223*L/18,   1279/24.0,  -1681*L/144.0],
        [223*L/18,  71*L2/27.0, 1681*L/144, -1097*L2/432.0],
        [1279/24.0, 1681*L/144, 59,         -223*L/18.0],
        [-1681*L/144, -1097*L2/432, -223*L/18, 71*L2/27.0]
    ])
    K = K - w2**2 * M2
    return K


def _torsion_1d_taylor(G, J, rho, Ix, L, omega):
    """Taylor de K para torção 1D."""
    K0, M1 = torsion_1d_K0_M1(G, J, rho, Ix, L)
    w2 = omega**2
    return K0 - w2 * M1


# ============================================================================
# Matrizes K0 (rigidez estática) e M1 (massa consistente) — ANALÍTICAS
# ============================================================================

def truss_1d_K0_M1(E, A, rho, L):
    K0 = (E * A / L) * np.array([[1, -1], [-1, 1]])
    M1 = (rho * A * L / 6.0) * np.array([[2, 1], [1, 2]])
    return K0, M1


def beam_1d_K0_M1(E, I, rho, A, L):
    L2 = L * L
    L3 = L2 * L
    K0 = (E * I / L3) * np.array([
        [12,    6*L,   -12,    6*L],
        [6*L,   4*L2,  -6*L,   2*L2],
        [-12,   -6*L,  12,     -6*L],
        [6*L,   2*L2,  -6*L,   4*L2]
    ])
    m = rho * A
    M1 = (m * L / 420.0) * np.array([
        [156,    22*L,   54,     -13*L],
        [22*L,   4*L2,   13*L,   -3*L2],
        [54,     13*L,   156,    -22*L],
        [-13*L,  -3*L2,  -22*L,  4*L2]
    ])
    return K0, M1


def torsion_1d_K0_M1(G, J, rho, Ix, L):
    K0 = (G * J / L) * np.array([[1, -1], [-1, 1]])
    Im = rho * Ix
    M1 = (Im * L / 6.0) * np.array([[2, 1], [1, 2]])
    return K0, M1


def truss_1d_M2(E, A, rho, L):
    """M2 analítica para treliça 1D (da expansão em série, Eq. 4-15)."""
    coeff = rho**2 * A * L**3 / (360.0 * E)
    return coeff * np.array([[8, 7], [7, 8]])


def beam_1d_M2(E, I, rho, A, L):
    """M2 analítica para viga 1D (Eq. 4-33)."""
    m = rho * A
    L2 = L * L
    coeff = m**2 * L**5 / (161700.0 * E * I)
    return coeff * np.array([
        [59.0,           223.0*L/18.0,    1279.0/24.0,    -1681.0*L/144.0],
        [223.0*L/18.0,   71.0*L2/27.0,    1681.0*L/144.0, -1097.0*L2/432.0],
        [1279.0/24.0,    1681.0*L/144.0,  59.0,           -223.0*L/18.0],
        [-1681.0*L/144.0,-1097.0*L2/432.0,-223.0*L/18.0,  71.0*L2/27.0]
    ])


def torsion_1d_M2(G, J, rho, Ix, L):
    """M2 analítica para torção 1D."""
    Im = rho * Ix
    coeff = Im**2 * L**3 / (360.0 * G * J)
    return coeff * np.array([[8, 7], [7, 8]])


def beam_2d_K0_M1(E, A, I, rho, L):
    """K0 e M1 analíticos para viga 2D (6 GDL)."""
    K0 = np.zeros((6, 6))
    M1 = np.zeros((6, 6))

    Kt0, Mt1 = truss_1d_K0_M1(E, A, rho, L)
    K0[0, 0] = Kt0[0, 0]; K0[0, 3] = Kt0[0, 1]
    K0[3, 0] = Kt0[1, 0]; K0[3, 3] = Kt0[1, 1]
    M1[0, 0] = Mt1[0, 0]; M1[0, 3] = Mt1[0, 1]
    M1[3, 0] = Mt1[1, 0]; M1[3, 3] = Mt1[1, 1]

    Kb0, Mb1 = beam_1d_K0_M1(E, I, rho, A, L)
    idx = [1, 2, 4, 5]
    for i_l, i_g in enumerate(idx):
        for j_l, j_g in enumerate(idx):
            K0[i_g, j_g] = Kb0[i_l, j_l]
            M1[i_g, j_g] = Mb1[i_l, j_l]

    return K0, M1


def truss_2d_K0_M1(E, A, I, rho, L):
    """K0 e M1 para treliça 2D (4 GDL) — condensação estática dos GDL rotacionais."""
    K6_0, M6_1 = beam_2d_K0_M1(E, A, I, rho, L)

    keep = [0, 1, 3, 4]
    rot = [2, 5]

    Kdd = K6_0[np.ix_(keep, keep)]
    Kdr = K6_0[np.ix_(keep, rot)]
    Krr = K6_0[np.ix_(rot, rot)]
    Krd = K6_0[np.ix_(rot, keep)]

    Mdd = M6_1[np.ix_(keep, keep)]
    Mdr = M6_1[np.ix_(keep, rot)]
    Mrd = M6_1[np.ix_(rot, keep)]
    Mrr = M6_1[np.ix_(rot, rot)]

    Krr_inv = np.linalg.inv(Krr)
    K0 = Kdd - Kdr @ Krr_inv @ Krd
    M1 = (Mdd - Kdr @ Krr_inv @ Mrd
           - Mdr @ Krr_inv @ Krd
           + Kdr @ Krr_inv @ Mrr @ Krr_inv @ Krd)

    return K0, M1


def beam_3d_K0_M1(E, G, A, Iy, Iz, J, Ix, rho, L):
    """K0 e M1 analíticos para viga 3D (12 GDL)."""
    K0 = np.zeros((12, 12))
    M1 = np.zeros((12, 12))

    # Treliça (axial) → GDL 0, 6
    Kt0, Mt1 = truss_1d_K0_M1(E, A, rho, L)
    for i, ig in enumerate([0, 6]):
        for j, jg in enumerate([0, 6]):
            K0[ig, jg] = Kt0[i, j]
            M1[ig, jg] = Mt1[i, j]

    # Viga no plano x-y (Iz) → GDL 1, 5, 7, 11
    Kbz0, Mbz1 = beam_1d_K0_M1(E, Iz, rho, A, L)
    idx_bz = [1, 5, 7, 11]
    for i_l, i_g in enumerate(idx_bz):
        for j_l, j_g in enumerate(idx_bz):
            K0[i_g, j_g] = Kbz0[i_l, j_l]
            M1[i_g, j_g] = Mbz1[i_l, j_l]

    # Viga no plano x-z (Iy) → GDL 2, 4, 8, 10 (com sinais ajustados)
    Kby0, Mby1 = beam_1d_K0_M1(E, Iy, rho, A, L)
    idx_by = [2, 4, 8, 10]
    sign_by = [1, -1, 1, -1]
    for i_l in range(4):
        for j_l in range(4):
            K0[idx_by[i_l], idx_by[j_l]] = sign_by[i_l] * sign_by[j_l] * Kby0[i_l, j_l]
            M1[idx_by[i_l], idx_by[j_l]] = sign_by[i_l] * sign_by[j_l] * Mby1[i_l, j_l]

    # Torção → GDL 3, 9
    Ktor0, Mtor1 = torsion_1d_K0_M1(G, J, rho, Ix, L)
    for i, ig in enumerate([3, 9]):
        for j, jg in enumerate([3, 9]):
            K0[ig, jg] = Ktor0[i, j]
            M1[ig, jg] = Mtor1[i, j]

    return K0, M1


def beam_2d_M2(E, A, I, rho, L):
    """M2 analítica para viga 2D (6 GDL)."""
    M2 = np.zeros((6, 6))
    Mt2 = truss_1d_M2(E, A, rho, L)
    M2[0, 0] = Mt2[0, 0]; M2[0, 3] = Mt2[0, 1]
    M2[3, 0] = Mt2[1, 0]; M2[3, 3] = Mt2[1, 1]
    Mb2 = beam_1d_M2(E, I, rho, A, L)
    idx = [1, 2, 4, 5]
    for i_l, i_g in enumerate(idx):
        for j_l, j_g in enumerate(idx):
            M2[i_g, j_g] = Mb2[i_l, j_l]
    return M2


def truss_2d_M2(E, A, I, rho, L):
    """M2 para treliça 2D (4 GDL) — extraída de K(ω) condensada."""
    K0, M1 = truss_2d_K0_M1(E, A, I, rho, L)
    K_func = lambda w: truss_2d_stiffness(E, A, I, rho, L, w)
    return _extract_M2_from_K_omega(K_func, K0, M1, E, rho, L)


def beam_3d_M2(E, G, A, Iy, Iz, J, Ix, rho, L):
    """M2 analítica para viga 3D (12 GDL)."""
    M2 = np.zeros((12, 12))
    Mt2 = truss_1d_M2(E, A, rho, L)
    for i, ig in enumerate([0, 6]):
        for j, jg in enumerate([0, 6]):
            M2[ig, jg] = Mt2[i, j]
    Mbz2 = beam_1d_M2(E, Iz, rho, A, L)
    idx_bz = [1, 5, 7, 11]
    for i_l, i_g in enumerate(idx_bz):
        for j_l, j_g in enumerate(idx_bz):
            M2[i_g, j_g] = Mbz2[i_l, j_l]
    Mby2 = beam_1d_M2(E, Iy, rho, A, L)
    idx_by = [2, 4, 8, 10]
    sign_by = [1, -1, 1, -1]
    for i_l in range(4):
        for j_l in range(4):
            M2[idx_by[i_l], idx_by[j_l]] = sign_by[i_l]*sign_by[j_l]*Mby2[i_l, j_l]
    Mtor2 = torsion_1d_M2(G, J, rho, Ix, L)
    for i, ig in enumerate([3, 9]):
        for j, jg in enumerate([3, 9]):
            M2[ig, jg] = Mtor2[i, j]
    return M2


def truss_3d_M2(E, G, A, Iy, Iz, J, Ix, rho, L):
    """M2 para treliça 3D (6 GDL) — extraída de K(ω) condensada."""
    K0, M1 = truss_3d_K0_M1(E, G, A, Iy, Iz, J, Ix, rho, L)
    K_func = lambda w: truss_3d_stiffness(E, G, A, Iy, Iz, J, Ix, rho, L, w)
    return _extract_M2_from_K_omega(K_func, K0, M1, E, rho, L)


def _extract_M2_from_K_omega(K_func, K0, M1, E, rho, L):
    """
    Extract M2 numerically from K(ω) using Richardson extrapolation.
    Uses very small ω to minimize contamination from higher-order terms.
    """
    w_base = 0.01 * np.sqrt(E / rho) / L
    estimates = []
    for factor in [1.0, 0.5, 0.25]:
        w = w_base * factor
        K_w = K_func(w)
        w2 = w**2
        residual = K0 - K_w - w2 * M1
        M2_est = residual / w2**2
        estimates.append(M2_est)
    M2 = estimates[0]
    if len(estimates) >= 3:
        M2 = (4 * estimates[1] - estimates[0]) / 3.0
    M2 = 0.5 * (M2 + M2.T)
    return M2


def _condense_beam_1d_rot(K0_4, M1_4):
    """Condensa GDL rotacionais de uma viga 1D 4×4 → 2×2 translacional."""
    kv = [0, 2]
    kr = [1, 3]
    K0dd = K0_4[np.ix_(kv, kv)]
    K0dr = K0_4[np.ix_(kv, kr)]
    K0rr = K0_4[np.ix_(kr, kr)]
    K0rd = K0_4[np.ix_(kr, kv)]
    M1dd = M1_4[np.ix_(kv, kv)]
    M1dr = M1_4[np.ix_(kv, kr)]
    M1rd = M1_4[np.ix_(kr, kv)]
    M1rr = M1_4[np.ix_(kr, kr)]
    Ki = np.linalg.inv(K0rr)
    K0c = K0dd - K0dr @ Ki @ K0rd
    M1c = (M1dd - K0dr @ Ki @ M1rd
           - M1dr @ Ki @ K0rd
           + K0dr @ Ki @ M1rr @ Ki @ K0rd)
    return K0c, M1c


def _condense_beam_1d_Kw(K4):
    """Condensa GDL rotacionais de uma viga 1D K(ω) 4×4 → 2×2."""
    kv = [0, 2]
    kr = [1, 3]
    Kdd = K4[np.ix_(kv, kv)]
    Kdr = K4[np.ix_(kv, kr)]
    Krr = K4[np.ix_(kr, kr)]
    Krd = K4[np.ix_(kr, kv)]
    return Kdd - Kdr @ np.linalg.solve(Krr, Krd)


def truss_3d_K0_M1(E, G, A, Iy, Iz, J, Ix, rho, L):
    """K0 e M1 para treliça 3D (6 GDL) — condensação por componente 1D."""
    K0 = np.zeros((6, 6))
    M1 = np.zeros((6, 6))

    Kt0, Mt1 = truss_1d_K0_M1(E, A, rho, L)
    for i, ig in enumerate([0, 3]):
        for j, jg in enumerate([0, 3]):
            K0[ig, jg] = Kt0[i, j]
            M1[ig, jg] = Mt1[i, j]

    Kbz0, Mbz1 = beam_1d_K0_M1(E, Iz, rho, A, L)
    K0cz, M1cz = _condense_beam_1d_rot(Kbz0, Mbz1)
    for i, ig in enumerate([1, 4]):
        for j, jg in enumerate([1, 4]):
            K0[ig, jg] = K0cz[i, j]
            M1[ig, jg] = M1cz[i, j]

    Kby0, Mby1 = beam_1d_K0_M1(E, Iy, rho, A, L)
    K0cy, M1cy = _condense_beam_1d_rot(Kby0, Mby1)
    for i, ig in enumerate([2, 5]):
        for j, jg in enumerate([2, 5]):
            K0[ig, jg] = K0cy[i, j]
            M1[ig, jg] = M1cy[i, j]

    return K0, M1


# ============================================================================
# Funções K(ω) para elementos compostos 2D/3D
# ============================================================================

def beam_2d_stiffness(E, A, I, rho, L, omega):
    """Rigidez efetiva 6x6 do elemento de viga 2D."""
    K = np.zeros((6, 6))
    Kt = truss_stiffness_1d(E, A, rho, L, omega)
    K[0, 0] = Kt[0, 0]; K[0, 3] = Kt[0, 1]
    K[3, 0] = Kt[1, 0]; K[3, 3] = Kt[1, 1]

    Kb = beam_stiffness_1d(E, I, rho, A, L, omega)
    idx_b = [1, 2, 4, 5]
    for i_l, i_g in enumerate(idx_b):
        for j_l, j_g in enumerate(idx_b):
            K[i_g, j_g] = Kb[i_l, j_l]
    return K


def truss_2d_stiffness(E, A, I, rho, L, omega):
    """Rigidez efetiva 4x4 do elemento de treliça 2D (condensação de rotações)."""
    K6 = beam_2d_stiffness(E, A, I, rho, L, omega)
    keep = [0, 1, 3, 4]
    rot = [2, 5]
    Kdd = K6[np.ix_(keep, keep)]
    Kdr = K6[np.ix_(keep, rot)]
    Krr = K6[np.ix_(rot, rot)]
    Krd = K6[np.ix_(rot, keep)]
    try:
        return Kdd - Kdr @ np.linalg.solve(Krr, Krd)
    except np.linalg.LinAlgError:
        return Kdd


def beam_3d_stiffness(E, G, A, Iy, Iz, J, Ix, rho, L, omega):
    """Rigidez efetiva 12x12 do elemento de viga 3D."""
    K = np.zeros((12, 12))
    Kt = truss_stiffness_1d(E, A, rho, L, omega)
    K[0, 0] = Kt[0, 0]; K[0, 6] = Kt[0, 1]
    K[6, 0] = Kt[1, 0]; K[6, 6] = Kt[1, 1]

    Kbz = beam_stiffness_1d(E, Iz, rho, A, L, omega)
    idx_bz = [1, 5, 7, 11]
    for i_l, i_g in enumerate(idx_bz):
        for j_l, j_g in enumerate(idx_bz):
            K[i_g, j_g] = Kbz[i_l, j_l]

    Kby = beam_stiffness_1d(E, Iy, rho, A, L, omega)
    idx_by = [2, 4, 8, 10]
    sign_by = [1, -1, 1, -1]
    for i_l in range(4):
        for j_l in range(4):
            K[idx_by[i_l], idx_by[j_l]] = sign_by[i_l] * sign_by[j_l] * Kby[i_l, j_l]

    Ktor = torsion_stiffness_1d(G, J, rho, Ix, L, omega)
    K[3, 3] = Ktor[0, 0]; K[3, 9] = Ktor[0, 1]
    K[9, 3] = Ktor[1, 0]; K[9, 9] = Ktor[1, 1]
    return K


def truss_3d_stiffness(E, G, A, Iy, Iz, J, Ix, rho, L, omega):
    """Rigidez efetiva 6x6 da treliça 3D — condensação por componente 1D."""
    K = np.zeros((6, 6))

    Kt = truss_stiffness_1d(E, A, rho, L, omega)
    for i, ig in enumerate([0, 3]):
        for j, jg in enumerate([0, 3]):
            K[ig, jg] = Kt[i, j]

    Kbz = beam_stiffness_1d(E, Iz, rho, A, L, omega)
    Kcz = _condense_beam_1d_Kw(Kbz)
    for i, ig in enumerate([1, 4]):
        for j, jg in enumerate([1, 4]):
            K[ig, jg] = Kcz[i, j]

    Kby = beam_stiffness_1d(E, Iy, rho, A, L, omega)
    Kcy = _condense_beam_1d_Kw(Kby)
    for i, ig in enumerate([2, 5]):
        for j, jg in enumerate([2, 5]):
            K[ig, jg] = Kcy[i, j]

    return K


# ============================================================================
# Matrizes de rotação e transformação
# ============================================================================

def rotation_matrix_2d(xi, yi, xj, yj):
    dx = xj - xi; dy = yj - yi
    L = sqrt(dx**2 + dy**2)
    c = dx / L; s = dy / L
    return np.array([
        [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]
    ])


def rotation_matrix_2d_truss(xi, yi, xj, yj):
    dx = xj - xi; dy = yj - yi
    L = sqrt(dx**2 + dy**2)
    c = dx / L; s = dy / L
    return np.array([
        [c, s, 0, 0], [-s, c, 0, 0],
        [0, 0, c, s], [0, 0, -s, c]
    ])


def rotation_matrix_3d(xi, yi, zi, xj, yj, zj):
    dx = xj - xi; dy = yj - yi; dz = zj - zi
    L = sqrt(dx**2 + dy**2 + dz**2)
    Cx = dx / L; Cy = dy / L; Cz = dz / L
    Lp = sqrt(Cx**2 + Cz**2)
    if Lp < 1e-10:
        return np.array([[0, Cy, 0], [-Cy, 0, 0], [0, 0, 1]])
    return np.array([
        [Cx, Cy, Cz],
        [-Cx*Cy/Lp, Lp, -Cy*Cz/Lp],
        [-Cz/Lp, 0, Cx/Lp]
    ])


def transform_matrix(K_local, T):
    """Transforma K local para global: Kg = T^T K T."""
    return T.T @ K_local @ T


def build_T_3d_beam(R):
    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*i+3, 3*i:3*i+3] = R
    return T


def build_T_3d_truss(R):
    T = np.zeros((6, 6))
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    return T
