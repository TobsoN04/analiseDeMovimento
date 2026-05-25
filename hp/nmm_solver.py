"""
hp.nmm_solver — Suporte para 3MM, 4MM, 5MM, 6MM.

Extrai M3, M4, M5, M6 analíticos via série de Taylor da K(ω) fechada,
e resolve o problema polinomial de grau n via companion estendido.
"""
from mpmath import mp, mpf, mpc, matrix, zeros, eye, eig, taylor, sqrt, sin, cos, sinh, cosh, exp
from hp.core import mp_inv, mp_solve, Timer


# =============================================================================
# Extração de séries M_j para elementos 1D
# =============================================================================

def truss_1d_M_series(E, A, rho, L, n_terms=6):
    """
    Para a treliça 1D K(ω) = (k EA/sin(kL))·[[cos(kL), -1],[-1, cos(kL)]]
    com k²=ω²ρ/E, retorna [M1, M2, ..., M_{n_terms}] em mpmath.matrix 2×2.

    Usa expansão de Taylor em u = ω²·ρ·L²/E:
        K_ij(u)/(EA/L) = sum c^{ij}_k · u^k
        M_j[i,j] = -c^{ij}_j · EA/L · (ρL²/E)^j
    """
    E_m = mpf(E); A_m = mpf(A); rho_m = mpf(rho); L_m = mpf(L)
    scale = E_m * A_m / L_m
    factor_per_step = rho_m * L_m * L_m / E_m  # (ρL²/E)

    # Função adimensional K_norm(u) / (EA/L)
    # diag: sqrt(u)·cot(sqrt(u))
    # off:  -sqrt(u)·csc(sqrt(u))
    # Para u=0 ambos têm valor 1 e -1 (limite)
    def diag(u):
        if u == 0:
            return mpf(1)
        s = sqrt(u)
        return s * cos(s) / sin(s)

    def off(u):
        if u == 0:
            return mpf(-1)
        s = sqrt(u)
        return -s / sin(s)

    c_diag = taylor(diag, 0, n_terms)  # [c_0, c_1, ..., c_n]
    c_off = taylor(off, 0, n_terms)

    M_list = []
    for j in range(1, n_terms + 1):
        # M_j = -c_j · scale · factor^j
        d = -c_diag[j] * scale * factor_per_step**j
        o = -c_off[j] * scale * factor_per_step**j
        M_j = matrix([[d, o], [o, d]])
        M_list.append(M_j)
    return M_list


def torsion_1d_M_series(G, J, rho, Ix, L, n_terms=6):
    """Idem treliça mas com G·J no lugar de E·A, e Ix·ρ no lugar de A·ρ."""
    G_m = mpf(G); J_m = mpf(J); rho_m = mpf(rho); Ix_m = mpf(Ix); L_m = mpf(L)
    scale = G_m * J_m / L_m
    factor = rho_m * Ix_m * L_m * L_m / (G_m * J_m)
    # K(ω) torsão tem mesma forma da treliça com (k²=ω²·ρIx/(GJ))
    def diag(u):
        if u == 0:
            return mpf(1)
        s = sqrt(u)
        return s * cos(s) / sin(s)
    def off(u):
        if u == 0:
            return mpf(-1)
        s = sqrt(u)
        return -s / sin(s)
    c_diag = taylor(diag, 0, n_terms)
    c_off = taylor(off, 0, n_terms)
    M_list = []
    for j in range(1, n_terms + 1):
        d = -c_diag[j] * scale * factor**j
        o = -c_off[j] * scale * factor**j
        M_list.append(matrix([[d, o], [o, d]]))
    return M_list


def beam_1d_M_series(E, I, rho, A, L, n_terms=6):
    """
    Para a viga K(ω) = (EI/(γL³))·[matriz com a_ij(α)] onde α = βL,
    β⁴ = ω²·ρA/(EI).
    Usa Taylor em v = α⁴ = ω²·ρAL⁴/(EI).

    K_ij(v)/(EI/L³) = f_ij(v)
    M_j[i,k] = -coef_j de f_ij · EI/L³ · (ρAL⁴/EI)^j = -coef_j · EI/L³ · factor^j
    onde factor = ρAL⁴/(EI).
    """
    E_m = mpf(E); I_m = mpf(I); rho_m = mpf(rho); A_m = mpf(A); L_m = mpf(L)
    scale = E_m * I_m / L_m**3
    factor = rho_m * A_m * L_m**4 / (E_m * I_m)

    def alpha(v):
        if v == 0:
            return mpf(0)
        return v**mpf('0.25')

    def f_diag_11(v):
        """K[0,0] normalizado: a11/(gamma) = α³(Sc+Cs)/(1-cC)."""
        if v == 0:
            return mpf(12)  # K0[0,0] = 12 EI/L³
        a = alpha(v)
        c, C, s, S = cos(a), cosh(a), sin(a), sinh(a)
        gamma = mpf(1) - c*C
        return (a**3 * (S*c + C*s)) / gamma

    def f_22(v):
        """K[1,1]/(EI/L) = a22/gamma, multiplicar por L²/L → diferente escala."""
        if v == 0:
            return mpf(4)
        a = alpha(v)
        c, C, s, S = cos(a), cosh(a), sin(a), sinh(a)
        gamma = mpf(1) - c*C
        # a22 = α·L²·(C·s - S·c) → escalado por EI/(γL³): EI/L · (C·s - S·c)·α/γ
        return a * (C*s - S*c) / gamma

    def f_12(v):
        if v == 0:
            return mpf(6)  # K0[0,1]/L = 6 EI/L²
        a = alpha(v)
        c, C, s, S = cos(a), cosh(a), sin(a), sinh(a)
        gamma = mpf(1) - c*C
        return a**2 * S * s / gamma

    def f_13(v):
        if v == 0:
            return mpf(-12)
        a = alpha(v)
        c, C, s, S = cos(a), cosh(a), sin(a), sinh(a)
        gamma = mpf(1) - c*C
        return -a**3 * (s + S) / gamma

    def f_14(v):
        if v == 0:
            return mpf(6)
        a = alpha(v)
        c, C, s, S = cos(a), cosh(a), sin(a), sinh(a)
        gamma = mpf(1) - c*C
        return a**2 * (C - c) / gamma

    def f_24(v):
        if v == 0:
            return mpf(2)
        a = alpha(v)
        c, C, s, S = cos(a), cosh(a), sin(a), sinh(a)
        gamma = mpf(1) - c*C
        return a * (S - s) / gamma

    # Taylor coefficients
    c11 = taylor(f_diag_11, 0, n_terms)  # K[0,0] e K[2,2]
    c22 = taylor(f_22, 0, n_terms)        # K[1,1] e K[3,3]
    c12 = taylor(f_12, 0, n_terms)        # K[0,1] e K[2,3] (com L)
    c13 = taylor(f_13, 0, n_terms)
    c14 = taylor(f_14, 0, n_terms)        # K[0,3] e K[2,1] (com L)
    c24 = taylor(f_24, 0, n_terms)        # K[1,3] (com L²)

    M_list = []
    for j in range(1, n_terms + 1):
        # M_j[i,k] = -c_j_ik · scale · L^(grau_L) · factor^j
        m11 = -c11[j] * scale * factor**j
        m22 = -c22[j] * scale * L_m**2 * factor**j
        m12 = -c12[j] * scale * L_m * factor**j
        m13 = -c13[j] * scale * factor**j
        m14 = -c14[j] * scale * L_m * factor**j
        m24 = -c24[j] * scale * L_m**2 * factor**j
        # montar matriz simétrica 4x4 com padrão Williams-Wittrick:
        # [[a11, a12, a13, a14],[a12, a22, -a14, a24],[a13,-a14, a11,-a12],[a14, a24,-a12, a22]]
        M_j = matrix([
            [m11,  m12,  m13,  m14],
            [m12,  m22, -m14,  m24],
            [m13, -m14,  m11, -m12],
            [m14,  m24, -m12,  m22]
        ])
        M_list.append(M_j)
    return M_list


# =============================================================================
# Composite elements (2D / 3D)
# =============================================================================

def _embed(M_global, M_block, idx):
    for i, ig in enumerate(idx):
        for j, jg in enumerate(idx):
            M_global[ig, jg] = M_global[ig, jg] + M_block[i, j]
    return M_global


def beam_2d_M_series(E, A, I, rho, L, n_terms=6):
    """Série M_j para viga 2D (6 GDL)."""
    Mt_list = truss_1d_M_series(E, A, rho, L, n_terms)
    Mb_list = beam_1d_M_series(E, I, rho, A, L, n_terms)
    out = []
    for j in range(n_terms):
        M = zeros(6, 6)
        _embed(M, Mt_list[j], [0, 3])
        _embed(M, Mb_list[j], [1, 2, 4, 5])
        out.append(M)
    return out


def beam_3d_M_series(E, G, A, Iy, Iz, J, Ix, rho, L, n_terms=6):
    """Série M_j para viga 3D (12 GDL)."""
    Mt = truss_1d_M_series(E, A, rho, L, n_terms)
    Mbz = beam_1d_M_series(E, Iz, rho, A, L, n_terms)
    Mby = beam_1d_M_series(E, Iy, rho, A, L, n_terms)
    Mtor = torsion_1d_M_series(G, J, rho, Ix, L, n_terms)
    out = []
    sign = [1, -1, 1, -1]
    for j in range(n_terms):
        M = zeros(12, 12)
        _embed(M, Mt[j], [0, 6])
        _embed(M, Mbz[j], [1, 5, 7, 11])
        idx_by = [2, 4, 8, 10]
        for i_l in range(4):
            for j_l in range(4):
                M[idx_by[i_l], idx_by[j_l]] = M[idx_by[i_l], idx_by[j_l]] + sign[i_l]*sign[j_l]*Mby[j][i_l, j_l]
        _embed(M, Mtor[j], [3, 9])
        out.append(M)
    return out


def _condense_static_series(K0_4, M_4_list):
    """Condensa rotações de uma matriz 4x4 série [M1, M2, ...] usando K0."""
    kv = [0, 2]
    kr = [1, 3]
    def blk(M, r, c):
        out = zeros(len(r), len(c))
        for i, ir in enumerate(r):
            for j, jc in enumerate(c):
                out[i, j] = M[ir, jc]
        return out
    K0dd = blk(K0_4, kv, kv); K0dr = blk(K0_4, kv, kr)
    K0rr = blk(K0_4, kr, kr); K0rd = blk(K0_4, kr, kv)
    Ki = mp_inv(K0rr)
    K0c = K0dd - K0dr * Ki * K0rd
    out = [K0c]
    for M in M_4_list:
        Mdd = blk(M, kv, kv); Mdr = blk(M, kv, kr)
        Mrr = blk(M, kr, kr); Mrd = blk(M, kr, kv)
        Mc = (Mdd - K0dr*Ki*Mrd - Mdr*Ki*K0rd + K0dr*Ki*Mrr*Ki*K0rd)
        out.append(Mc)
    return out  # [K0_cond, M1_cond, M2_cond, ...]


def truss_2d_M_series(E, A, I, rho, L, n_terms=6):
    """Série M_j para treliça 2D (4 GDL) via condensação."""
    from hp.core import beam_1d_K0_M1
    K0_4, _ = beam_1d_K0_M1(E, I, rho, A, L)
    M_b_list = beam_1d_M_series(E, I, rho, A, L, n_terms)
    cond = _condense_static_series(K0_4, M_b_list)
    # cond[0] é K0_cond ; cond[1..n] são M_j cond
    # Resultado precisa ir em 4x4 com axial + transversal:
    Mt_list = truss_1d_M_series(E, A, rho, L, n_terms)
    out = []
    for j in range(n_terms):
        M = zeros(4, 4)
        # axial em [0, 2]
        _embed(M, Mt_list[j], [0, 2])
        # transversal condensado em [1, 3]
        _embed(M, cond[j+1], [1, 3])
        out.append(M)
    return out


def truss_3d_M_series(E, G, A, Iy, Iz, J, Ix, rho, L, n_terms=6):
    """Série M_j para treliça 3D (6 GDL) via condensação por componente."""
    from hp.core import beam_1d_K0_M1
    Mt = truss_1d_M_series(E, A, rho, L, n_terms)
    Kbz0, _ = beam_1d_K0_M1(E, Iz, rho, A, L)
    Mbz = beam_1d_M_series(E, Iz, rho, A, L, n_terms)
    cond_z = _condense_static_series(Kbz0, Mbz)
    Kby0, _ = beam_1d_K0_M1(E, Iy, rho, A, L)
    Mby = beam_1d_M_series(E, Iy, rho, A, L, n_terms)
    cond_y = _condense_static_series(Kby0, Mby)
    out = []
    for j in range(n_terms):
        M = zeros(6, 6)
        _embed(M, Mt[j], [0, 3])
        _embed(M, cond_z[j+1], [1, 4])
        _embed(M, cond_y[j+1], [2, 5])
        out.append(M)
    return out


# =============================================================================
# Solver polinomial companion para grau n (nMM)
# =============================================================================

def solve_nmm_companion(K0, M_list):
    """
    Problema: (K0 - λM1 - λ²M2 - ... - λⁿ Mₙ) Φ = 0,  λ = ω².

    Linearização companion (forma standard para autovalor polinomial):
        A z = λ B z,  z = [Φ, λΦ, λ²Φ, ..., λⁿ⁻¹Φ]ᵀ
        A = [[ 0     I     0   ...  0  ]
             [ 0     0     I   ...  0  ]
             [...                     ...]
             [ 0     0     0   ...  I  ]
             [ K0   -M1   -M2  ...  -M_{n-1} ]]
        B = block_diag(I, I, ..., I, M_n)

    Multiplicar a última linha por λ recupera:
        K0·Φ - M1·λΦ - M2·λ²Φ - ... - M_{n-1}·λⁿ⁻¹Φ = λⁿ·M_n·Φ
        ⇔ K0·Φ = λM1·Φ + λ²M2·Φ + ... + λⁿMn·Φ.

    Retorna lista de ω = sqrt(λ_real_positivo) ordenada.
    """
    n_size = K0.rows
    n_deg = len(M_list)  # número de matrizes de massa (M1, M2, ..., M_n)
    N = n_size * n_deg
    I_n = eye(n_size)
    Z_n = zeros(n_size, n_size)

    A = zeros(N, N)
    B = zeros(N, N)

    # Linhas 0..(n_deg-1) (exceto última): A[i, i+1·n_size] = I
    for k in range(n_deg - 1):
        for i in range(n_size):
            for j in range(n_size):
                A[k*n_size + i, (k+1)*n_size + j] = I_n[i, j]
                B[k*n_size + i, k*n_size + j] = I_n[i, j]

    # Última linha-bloco: [K0, -M1, -M2, ..., -M_{n-1}]
    last_row = (n_deg - 1) * n_size
    for i in range(n_size):
        for j in range(n_size):
            A[last_row + i, j] = K0[i, j]
    for k in range(1, n_deg):
        sign = -1
        for i in range(n_size):
            for j in range(n_size):
                A[last_row + i, k*n_size + j] = sign * M_list[k-1][i, j]
    # B: bloco diagonal final = M_n
    for i in range(n_size):
        for j in range(n_size):
            B[last_row + i, last_row + j] = M_list[n_deg - 1][i, j]

    Binv = mp_inv(B)
    C = Binv * A
    eigvals, eigvecs = eig(C)

    pairs = []
    for k, lam in enumerate(eigvals):
        if hasattr(lam, 'imag'):
            re, im = lam.real, lam.imag
        else:
            re, im = mpf(lam), mpf(0)
        # físico: λ real positivo
        if re > mpf('1e-15') and abs(im) < abs(re) * mpf('1e-20'):
            pairs.append((re, k))
    pairs.sort(key=lambda x: x[0])
    omegas = [sqrt(lam) for lam, _ in pairs]

    Phi = zeros(n_size, len(pairs))
    for col, (_, k) in enumerate(pairs):
        for i in range(n_size):
            v = eigvecs[i, k]
            Phi[i, col] = v.real if hasattr(v, 'real') else v
    return omegas, Phi


# =============================================================================
# Assembly genérico em série
# =============================================================================

def assemble_K0_M_series(struct, n_mm):
    """
    Monta K0 e [M1, M2, ..., M_{n_mm}] globais reduzidas.
    n_mm = número de matrizes de massa desejadas (1 a 6).
    """
    from hp.core import zeros as zeros_, transform
    n_total = len(struct.nodes) * struct.dof_per_node
    K0_full = zeros(n_total, n_total)
    M_full = [zeros(n_total, n_total) for _ in range(n_mm)]

    for e in struct.elements:
        L = e.L
        if struct.dim == '2d' and struct.elem_type == 'truss':
            from hp.core import truss_2d_K0_M1_M2, beam_2d_K0_M1_M2
            K0_loc, _, _ = truss_2d_K0_M1_M2(e.E, e.A, e.I, e.rho, L)
            M_locs = truss_2d_M_series(e.E, e.A, e.I, e.rho, L, n_mm)
        elif struct.dim == '2d' and struct.elem_type == 'frame':
            from hp.core import beam_2d_K0_M1_M2
            K0_loc, _, _ = beam_2d_K0_M1_M2(e.E, e.A, e.I, e.rho, L)
            M_locs = beam_2d_M_series(e.E, e.A, e.I, e.rho, L, n_mm)
        elif struct.dim == '3d' and struct.elem_type == 'truss':
            from hp.core import truss_3d_K0_M1_M2
            K0_loc, _, _ = truss_3d_K0_M1_M2(e.E, e.G, e.A, e.Iy, e.Iz, e.J, e.Ix, e.rho, L)
            M_locs = truss_3d_M_series(e.E, e.G, e.A, e.Iy, e.Iz, e.J, e.Ix, e.rho, L, n_mm)
        else:  # 3d frame
            from hp.core import beam_3d_K0_M1_M2
            K0_loc, _, _ = beam_3d_K0_M1_M2(e.E, e.G, e.A, e.Iy, e.Iz, e.J, e.Ix, e.rho, L)
            M_locs = beam_3d_M_series(e.E, e.G, e.A, e.Iy, e.Iz, e.J, e.Ix, e.rho, L, n_mm)

        T, edofs = struct._elem_T_dofs(e)
        K0g = transform(K0_loc, T)
        Mg_list = [transform(M, T) for M in M_locs]

        for il, ig in enumerate(edofs):
            for jl, jg in enumerate(edofs):
                K0_full[ig, jg] = K0_full[ig, jg] + K0g[il, jl]
                for k in range(n_mm):
                    M_full[k][ig, jg] = M_full[k][ig, jg] + Mg_list[k][il, jl]

    free = struct.free_dofs()
    from hp.core import _sub
    K0_red = _sub(K0_full, free)
    M_red = [_sub(M, free) for M in M_full]
    return K0_red, M_red


def solve_struct_nmm(struct, n_mm):
    """Pipeline completo: monta K0+Mn e resolve para n_mm matrizes de massa."""
    K0, M_list = assemble_K0_M_series(struct, n_mm)
    if n_mm == 1:
        from hp.core import solve_1mm
        return solve_1mm(K0, M_list[0])
    return solve_nmm_companion(K0, M_list)
