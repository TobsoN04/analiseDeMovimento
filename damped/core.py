"""
damped.core — Versão com amortecimento (paralela ao hp.core, NÃO substitui).

Sistema completo: M·x¨ + C·x˙ + K·x = F(t)

Implementa:
1. Amortecimento de Rayleigh: C = α·M + β·K
   onde α, β são calibrados com 2 frequências e razões de amortecimento.
2. Problema de autovalor quadrático: (λ²M + λC + K)·x = 0
   Linearização: [[0, I],[-M⁻¹K, -M⁻¹C]] · z = λ·z
   Autovalores complexos λ = -ζω ± iω√(1-ζ²)
3. Resposta no tempo: integração de Newmark-β.
4. Função de Resposta em Frequência (FRF).

Usa mpmath em alta precisão (50 dígitos), assim como hp.core.
"""
import time
from mpmath import mp, mpf, mpc, matrix, eye, zeros, lu_solve, eig, sqrt, exp, sin, cos, pi
mp.dps = 50

# Reusar utilitários de hp.core
from hp.core import mp_inv, mp_solve, transform, Timer


def rayleigh_damping(M, K, omega1, omega2, zeta1, zeta2):
    """
    Calcula coeficientes α, β do amortecimento de Rayleigh:
        C = α·M + β·K
    de modo que para duas frequências dadas (ω₁, ω₂) os fatores de
    amortecimento são ζ₁, ζ₂.

    Fórmula:
        ζᵢ = (α/(2·ωᵢ)) + (β·ωᵢ/2)
    Sistema 2x2:
        [1/ω₁  ω₁] [α]   [2ζ₁]
        [1/ω₂  ω₂] [β] = [2ζ₂]
    """
    omega1, omega2 = mpf(omega1), mpf(omega2)
    zeta1, zeta2 = mpf(zeta1), mpf(zeta2)
    det = (1/omega1) * omega2 - omega1 * (1/omega2)
    alpha = (2*zeta1*omega2 - 2*zeta2*omega1) / det
    beta = (2*zeta2/omega2 - 2*zeta1/omega1) / det
    # solução simplificada:
    A = matrix([[1/omega1, omega1],
                [1/omega2, omega2]])
    b = matrix([[2*zeta1], [2*zeta2]])
    Ainv = mp_inv(A)
    sol = Ainv * b
    alpha = sol[0, 0]
    beta = sol[1, 0]
    C = alpha * M + beta * K
    return C, alpha, beta


def solve_quadratic_eigenvalue(M, C, K):
    """
    Resolve (λ²M + λC + K)·x = 0 via linearização companion.

    Forma usada (state-space):
        [[0      I ]     [[I  0]
         [-K   -C ]] z = [[0  M]] · λ z

    Equivalente: M·ẍ + C·ẋ + K·x = 0,  com z = [x, λx]^T.

    Retorna pares (λ_complexo, vetor_x). Os modos físicos têm
    λ = -ζω ± iω·sqrt(1-ζ²) com Re(λ) < 0 (amortecimento positivo).
    """
    n = M.rows
    I_n = eye(n)
    Z = zeros(n, n)
    # Matriz A = [[0, I], [-K, -C]]
    A = zeros(2*n, 2*n)
    B = zeros(2*n, 2*n)
    for i in range(n):
        for j in range(n):
            A[i, n+j] = I_n[i, j]
            A[n+i, j] = -K[i, j]
            A[n+i, n+j] = -C[i, j]
            B[i, j] = I_n[i, j]
            B[n+i, n+j] = M[i, j]
    Binv = mp_inv(B)
    Cmat = Binv * A
    eigvals, eigvecs = eig(Cmat)

    pairs = []
    for k, lam in enumerate(eigvals):
        # Em alta precisão, lam pode ser mpc
        if hasattr(lam, 'imag'):
            re, im = lam.real, lam.imag
        else:
            re, im = mpf(lam), mpf(0)
        # Modo físico: Re(λ) ≤ 0
        if re <= mpf('1e-10'):
            omega_d = abs(im)
            omega_n = sqrt(re*re + im*im)
            if omega_n > mpf('1e-10'):
                zeta = -re / omega_n
                pairs.append((omega_n, omega_d, zeta, lam, k))
    # ordenar por frequência natural
    pairs.sort(key=lambda p: p[0])
    return pairs, eigvecs


def newmark_beta_response(M, C, K, F_t, t_array, x0=None, v0=None,
                          beta=mpf('0.25'), gamma=mpf('0.5')):
    """
    Integra M·ẍ + C·ẋ + K·x = F(t) pelo método de Newmark-β.
    beta=0.25, gamma=0.5 → método de aceleração média (incondicionalmente estável).

    F_t : função t -> vetor força (mpmath column matrix de tamanho n)
    t_array : lista de instantes mpf
    x0, v0 : condições iniciais (mpmath column matrix)

    Retorna matriz n×len(t_array) com histórico de deslocamento.
    """
    n = M.rows
    if x0 is None:
        x0 = zeros(n, 1)
    if v0 is None:
        v0 = zeros(n, 1)

    nt = len(t_array)
    x_hist = zeros(n, nt)
    v_hist = zeros(n, nt)
    a_hist = zeros(n, nt)

    # Condição inicial: M·a₀ = F(0) - C·v₀ - K·x₀
    F0 = F_t(t_array[0])
    Cv0 = C * v0
    Kx0 = K * x0
    rhs0 = zeros(n, 1)
    for i in range(n):
        rhs0[i, 0] = F0[i, 0] - Cv0[i, 0] - Kx0[i, 0]
    a0 = mp_solve(M, rhs0)

    for i in range(n):
        x_hist[i, 0] = x0[i, 0]
        v_hist[i, 0] = v0[i, 0]
        a_hist[i, 0] = a0[i, 0]

    # Iteração
    for k in range(1, nt):
        dt = t_array[k] - t_array[k-1]
        # K_eff = K + (γ/(β·dt))·C + (1/(β·dt²))·M
        c0 = mpf(1) / (beta * dt * dt)
        c1 = gamma / (beta * dt)
        K_eff = K + c1 * C + c0 * M
        # Predição
        x_p = zeros(n, 1); v_p = zeros(n, 1)
        for i in range(n):
            x_p[i, 0] = x_hist[i, k-1] + dt * v_hist[i, k-1] + (mpf('0.5') - beta) * dt * dt * a_hist[i, k-1]
            v_p[i, 0] = v_hist[i, k-1] + (mpf(1) - gamma) * dt * a_hist[i, k-1]
        # RHS efetivo
        Fk = F_t(t_array[k])
        Mxp = M * (c0 * (-x_p))  # contribuição da predição
        # mais direta: usando força e correções
        # F_eff = F_k + M·(c0·x_{k-1} + c0·dt·v_{k-1} + (c0·(0.5-β)·dt²)·a_{k-1})
        #       + C·(c1·x_{k-1} + ...)
        # ... aqui usaremos a formulação predição/correção clássica:
        delta_x = zeros(n, 1)
        rhs = zeros(n, 1)
        for i in range(n):
            rhs[i, 0] = Fk[i, 0]
        # resolver K_eff·x_k = F_k + M·(c0·x_p + ...)
        Mp = M * x_p
        Cp = C * v_p
        rhs2 = zeros(n, 1)
        for i in range(n):
            rhs2[i, 0] = Fk[i, 0] + c0 * Mp[i, 0] + c1 * Cp[i, 0]
        x_new = mp_solve(K_eff, rhs2)
        a_new = zeros(n, 1); v_new = zeros(n, 1)
        for i in range(n):
            a_new[i, 0] = c0 * (x_new[i, 0] - x_p[i, 0])
            v_new[i, 0] = v_p[i, 0] + gamma * dt * a_new[i, 0]
            x_hist[i, k] = x_new[i, 0]
            v_hist[i, k] = v_new[i, 0]
            a_hist[i, k] = a_new[i, 0]
    return x_hist, v_hist, a_hist


def damped_modal_frequencies(M, C, K):
    """
    Retorna lista de tuplas (ω_n, ω_d, ζ) para cada modo físico,
    ordenadas por frequência natural ω_n.
    """
    pairs, _ = solve_quadratic_eigenvalue(M, C, K)
    # Cada modo aparece duas vezes (conjugado), filtramos
    seen = []
    for omega_n, omega_d, zeta, lam, k in pairs:
        is_dup = False
        for s in seen:
            if abs(omega_n - s[0]) < omega_n * mpf('1e-20'):
                is_dup = True; break
        if not is_dup:
            seen.append((omega_n, omega_d, zeta))
    return seen
