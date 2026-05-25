"""
Solver para análise dinâmica pelo Método Híbrido dos Elementos Finitos.

Abordagens:
1. Clássica (1MM): K0·Φ = ω²·M1·Φ  (problema generalizado padrão)
2. Avançada (2MM): (K0 - λM1 - λ²M2)Φ = 0  (linearização por companheiro)
3. Busca direta: det(K(ω)) = 0  (varredura + bisseção em K(ω))

Baseado nos Capítulos 3-4 da dissertação de Rodrigo Barros (PUC-Rio, 2017).
"""

import numpy as np
from scipy.linalg import eigh, eig, null_space, ordqz
from scipy.optimize import brentq


def solve_eigenvalue_1mm(K0, M1, n_modes=None):
    """Problema generalizado padrão: K0·Φ = ω²·M1·Φ."""
    K0s = 0.5 * (K0 + K0.T)
    M1s = 0.5 * (M1 + M1.T)

    try:
        eigenvalues, eigenvectors = eigh(K0s, M1s)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = eig(K0s, M1s)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    idx = np.where(eigenvalues > 1e-6)[0]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    frequencies = np.sqrt(eigenvalues)

    if n_modes is not None and n_modes < len(frequencies):
        frequencies = frequencies[:n_modes]
        eigenvectors = eigenvectors[:, :n_modes]

    return frequencies, eigenvectors


def solve_eigenvalue_2mm(K0, M1, M2, n_modes=None):
    """
    Problema de autovalor polinomial quadrático: (K0 - λM1 - λ²M2)Φ = 0.

    Linearizado via companheiro (Eq. 3-11 da dissertação):
        A·z = λ·B·z
    onde λ = ω², A = [[0, I], [-K0, M1]], B = [[I, 0], [0, M2]].
    """
    n = K0.shape[0]
    I_n = np.eye(n)
    Z_n = np.zeros((n, n))

    A = np.block([
        [Z_n, I_n],
        [K0, -M1]
    ])
    B = np.block([
        [I_n, Z_n],
        [Z_n, M2]
    ])

    eigenvalues, vl, vr = eig(A, B, left=True, right=True)

    valid = []
    for i, lam in enumerate(eigenvalues):
        if np.isfinite(lam) and np.isreal(lam) and np.real(lam) > 1e-6:
            valid.append((np.real(lam), i))

    valid.sort(key=lambda x: x[0])

    if not valid:
        return np.array([]), np.zeros((n, 0))

    lambdas = np.array([v[0] for v in valid])
    frequencies = np.sqrt(lambdas)

    Phi = np.zeros((n, len(valid)))
    for j, (_, idx) in enumerate(valid):
        phi = np.real(vr[:n, idx])
        if np.linalg.norm(phi) < 1e-14:
            phi = np.real(vr[n:, idx])
        Phi[:, j] = phi

    if n_modes is not None and n_modes < len(frequencies):
        frequencies = frequencies[:n_modes]
        Phi = Phi[:, :n_modes]

    return frequencies, Phi


def solve_eigenvalue_direct(K_omega_func, n_dof, omega_max, n_modes=None, n_scan=500):
    """
    Encontra frequências naturais diretamente a partir de K(ω).

    Varre ω de 0 a omega_max, detectando onde o menor autovalor de K(ω)
    cruza zero (mudança de sinal), e usa bisseção para refinar.
    """
    omegas_scan = np.linspace(1e-3, omega_max, n_scan)
    min_eigs = np.zeros(n_scan)

    for i, w in enumerate(omegas_scan):
        K = K_omega_func(w)
        eigs = np.linalg.eigvalsh(K)
        min_eigs[i] = eigs[0]

    frequencies = []
    for i in range(n_scan - 1):
        if min_eigs[i] * min_eigs[i+1] < 0:
            try:
                w_root = brentq(lambda w: np.linalg.eigvalsh(K_omega_func(w))[0],
                                omegas_scan[i], omegas_scan[i+1],
                                xtol=1e-8, rtol=1e-10)
                frequencies.append(w_root)
            except ValueError:
                pass

        # Buscar também cruzamentos em outros autovalores
        if i == 0:
            continue
        K_curr = K_omega_func(omegas_scan[i])
        eigs_curr = np.sort(np.linalg.eigvalsh(K_curr))
        K_next = K_omega_func(omegas_scan[i+1])
        eigs_next = np.sort(np.linalg.eigvalsh(K_next))

        for k in range(1, min(n_dof, len(eigs_curr))):
            if eigs_curr[k] * eigs_next[k] < 0:
                try:
                    w_root = brentq(
                        lambda w: np.sort(np.linalg.eigvalsh(K_omega_func(w)))[k],
                        omegas_scan[i], omegas_scan[i+1],
                        xtol=1e-8, rtol=1e-10
                    )
                    if not any(abs(w_root - f) < 1e-4 for f in frequencies):
                        frequencies.append(w_root)
                except ValueError:
                    pass

    frequencies = np.sort(frequencies)

    if n_modes is not None and n_modes < len(frequencies):
        frequencies = frequencies[:n_modes]

    return np.array(frequencies)


def find_mode_shapes(K_omega_func, frequencies):
    """Calcula autovetores (modos) no kernel de K(ω) em cada frequência natural."""
    n = K_omega_func(0.01).shape[0]
    n_modes = len(frequencies)
    Phi = np.zeros((n, n_modes))

    for i, w in enumerate(frequencies):
        K = K_omega_func(w)
        ns = null_space(K, rcond=1e-6)
        if ns.shape[1] > 0:
            Phi[:, i] = ns[:, 0]
        else:
            eigs, vecs = np.linalg.eigh(K)
            idx_min = np.argmin(np.abs(eigs))
            Phi[:, i] = vecs[:, idx_min]

    return Phi


def compute_generalized_mass(K_omega_func, frequencies, Phi):
    """
    Calcula massa generalizada: m_i = -Φᵢᵀ · dK/d(ω²) · Φᵢ

    dK/d(ω²) é a derivada numérica de K em relação a ω².
    Como K(ω) = K0 - ω²M1 - ω⁴M2 - ...,
    dK/d(ω²) = -M1 - 2ω²M2 - 3ω⁴M3 - ...
    """
    n_modes = len(frequencies)
    Phi_norm = np.copy(Phi)

    for i in range(n_modes):
        w = frequencies[i]
        phi = Phi[:, i]

        dw = max(w * 1e-5, 1e-3)
        w2 = w**2
        K_plus = K_omega_func(np.sqrt(w2 + w * dw))
        K_minus = K_omega_func(np.sqrt(max(w2 - w * dw, 1e-10)))
        dK_dw2 = (K_plus - K_minus) / (2 * w * dw)

        m_gen = -phi @ dK_dw2 @ phi
        if abs(m_gen) > 1e-15:
            Phi_norm[:, i] = phi / np.sqrt(abs(m_gen))

    return Phi_norm


def solve_eigenvalue_augmented(K0, mass_matrices, n_modes=None):
    """
    Interface unificada:
      1 matriz  → eigh padrão
      2 matrizes → linearização por companheiro (quadrático)
    """
    if len(mass_matrices) == 0:
        return np.array([]), np.zeros((K0.shape[0], 0))
    if len(mass_matrices) == 1:
        return solve_eigenvalue_1mm(K0, mass_matrices[0], n_modes)
    return solve_eigenvalue_2mm(K0, mass_matrices[0], mass_matrices[1], n_modes)


def normalize_modes(Phi, M1):
    """Normaliza autovetores pela massa: Φᵀ M1 Φ = I."""
    n_modes = Phi.shape[1]
    Phi_norm = np.copy(Phi)
    for i in range(n_modes):
        phi = Phi_norm[:, i]
        modal_mass = phi @ M1 @ phi
        if abs(modal_mass) > 1e-15:
            Phi_norm[:, i] = phi / np.sqrt(abs(modal_mass))
    return Phi_norm


def modal_superposition_fast(frequencies, Phi, force_amplitude, time_func,
                             t_array, d0=None, v0=None):
    """
    Superposição modal avançada (Eq. 3-15) com integração passo-a-passo.
    Usa solução analítica de Duhamel para carga linear por partes.
    """
    n_dof = Phi.shape[0]
    n_modes = min(len(frequencies), Phi.shape[1])
    n_times = len(t_array)

    modal_force = Phi[:, :n_modes].T @ force_amplitude

    if d0 is None:
        d0 = np.zeros(n_dof)
    if v0 is None:
        v0 = np.zeros(n_dof)

    eta0 = Phi[:, :n_modes].T @ d0
    eta_dot0 = Phi[:, :n_modes].T @ v0

    eta_history = np.zeros((n_modes, n_times))

    for i in range(n_modes):
        wi = frequencies[i]
        fi = modal_force[i]

        if abs(wi) < 1e-10:
            continue

        eta = eta0[i]
        eta_dot = eta_dot0[i]

        for j in range(n_times):
            if j == 0:
                eta_history[i, j] = eta
                continue

            dt = t_array[j] - t_array[j-1]

            f_prev = fi * time_func(t_array[j-1])
            f_curr = fi * time_func(t_array[j])

            c = np.cos(wi * dt)
            s = np.sin(wi * dt)
            df = (f_curr - f_prev) / dt if abs(dt) > 1e-15 else 0.0

            eta_new = (eta * c + eta_dot / wi * s
                       + f_prev / wi**2 * (1 - c)
                       + df / wi**3 * (wi * dt - s))
            eta_dot = (-eta * wi * s + eta_dot * c
                       + f_prev / wi * s
                       + df / wi**2 * (1 - c))

            eta = eta_new
            eta_history[i, j] = eta

    return Phi[:, :n_modes] @ eta_history
