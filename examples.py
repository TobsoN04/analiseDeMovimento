"""
Exemplos numéricos da dissertação de Rodrigo Barros (Cap. 5).

Cada exemplo reproduz uma estrutura da literatura (Weaver, Paz, Petyt)
e aplica a técnica de superposição modal avançada com até 6 matrizes de massa.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assembly import Structure
from solver import (
    solve_eigenvalue_1mm, solve_eigenvalue_2mm,
    solve_eigenvalue_augmented,
    normalize_modes, modal_superposition_fast
)


def heaviside(t):
    return 1.0 if t >= 0 else 0.0


def _make_impulse(t1):
    t2 = 2 * t1
    def time_func(t):
        return ((t / t1) * heaviside(t) * heaviside(t1 - t)
                + (2 - t / t1) * heaviside(t - t1) * heaviside(t2 - t))
    return time_func


def _run_analysis(struct, n_mass_matrices, time_func, obs_node, obs_dof,
                  t_final, n_steps, title, ylabel, prefix, plot):
    """
    Executa análise com 1 a n_mass_matrices matrizes de massa.

    Para 1MM: problema generalizado padrão K0·Φ = ω²·M1·Φ
    Para 2MM: linearização por companheiro (K0 - λM1 - λ²M2)Φ = 0
    """
    results = {}
    f_vec = struct.get_force_vector()
    t_array = np.linspace(0, t_final, n_steps)
    dof_idx = struct.get_node_dof_in_reduced(obs_node, obs_dof)

    n_mm_max = min(n_mass_matrices, 2)

    for n_mm in range(1, n_mm_max + 1):
        label = f'{n_mm}MM'
        print(f"\n  Calculando com {n_mm} matriz(es) de massa...")

        matrices = struct.compute_mass_matrices(n_mm)
        K0 = matrices[0]
        Mlist = matrices[1:]

        freqs, Phi = solve_eigenvalue_augmented(K0, Mlist)

        n_print = min(3, len(freqs))
        if len(freqs) > 0:
            print(f"    Frequências (rad/s): {np.array2string(freqs[:n_print], precision=2)}")
        else:
            print(f"    Nenhuma frequência encontrada.")
            continue

        Phi_norm = normalize_modes(Phi, Mlist[0])
        d_hist = modal_superposition_fast(freqs, Phi_norm, f_vec, time_func, t_array)

        if dof_idx is not None:
            results[label] = (t_array, d_hist[dof_idx, :], freqs)

    if plot and results:
        _plot_responses(results, title, ylabel, prefix)
        _plot_frequencies(results, f"{title} - Frequências", prefix)

    return results


# ============================================================================
# EXEMPLOS
# ============================================================================

def example_01_truss_2d(n_mass_matrices=6, plot=True):
    print("=" * 70)
    print("EXEMPLO 01 - Treliça plana (Weaver)")
    print("=" * 70)

    # Geometria correta (Fig 5.1): triângulo retângulo 3-4-5
    E = 207e9; rho = 7850.0; L = 6.35
    A = 6.451e-3; I = 16960e-8
    x3 = 0.6 * L   # base = 0.6L
    y2 = 0.8 * L   # altura = 0.8L

    struct = Structure(dim='2d', elem_type='truss')
    struct.add_node(1, 0.0, 0.0)
    struct.add_node(2, x3, y2)
    struct.add_node(3, x3, 0.0)
    struct.add_element(1, 1, 2, E, A, rho, I=I)          # hipotenusa, área A
    struct.add_element(2, 3, 2, E, 0.8 * A, rho, I=I)    # vertical, área 0.8A
    struct.add_element(3, 1, 3, E, 0.6 * A, rho, I=I)    # base, área 0.6A
    struct.add_constraint(1, [1])      # nó 1: restrição vertical
    struct.add_constraint(3, [0, 1])   # nó 3: restrição total
    struct.add_force(2, 0, 90e3)       # P(t) horizontal no nó 2

    return _run_analysis(struct, n_mass_matrices, _make_impulse(3.75e-3),
                         2, 1, 0.02, 500,
                         "Exemplo 01 - Treliça Plana (Weaver)",
                         "Deslocamento do nó 2 (m)", "ex01", plot)


def example_03_frame_2d(n_mass_matrices=6, plot=True):
    print("=" * 70)
    print("EXEMPLO 03 - Pórtico bidimensional (Weaver)")
    print("=" * 70)

    E = 207e9; rho = 7850.0
    A = 51.61e-4; I = 4.346e-5; L = 3.048

    struct = Structure(dim='2d', elem_type='frame')
    struct.add_node(1, 0.0, 0.0)
    struct.add_node(2, L, 0.0)
    struct.add_node(3, 0.0, L)
    struct.add_node(4, L, L)
    struct.add_element(1, 1, 3, E, A, rho, I=I)
    struct.add_element(2, 2, 4, E, A, rho, I=I)
    struct.add_element(3, 3, 4, E, A, rho, I=I)
    struct.add_constraint(1, [0, 1, 2])
    struct.add_constraint(2, [0, 1, 2])
    struct.add_force(4, 0, 44.48e3)

    return _run_analysis(struct, n_mass_matrices, _make_impulse(3.0e-3),
                         4, 0, 0.02, 500,
                         "Exemplo 03 - Pórtico 2D (Weaver)",
                         "Deslocamento do nó 4 (m)", "ex03", plot)


def example_04_frame_3d(n_mass_matrices=6, plot=True):
    print("=" * 70)
    print("EXEMPLO 04 - Pórtico espacial (Paz)")
    print("=" * 70)

    E = 2.0e11; G = 0.8e11; rho = 7850.0
    A = 0.04; Iy = 1.333e-4; Iz = 1.333e-4
    J = 2.253e-4; Ix = 2.667e-4; L = 5.0

    struct = Structure(dim='3d', elem_type='frame')
    struct.add_node(1, L, 0.0, 0.0)
    struct.add_node(2, 0.0, 0.0, 0.0)
    struct.add_node(3, 0.0, L, 0.0)
    struct.add_node(4, 0.0, 0.0, L)
    struct.add_element(1, 2, 1, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_element(2, 2, 3, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_element(3, 2, 4, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_constraint(1, [0, 1, 2, 3, 4, 5])
    struct.add_constraint(3, [0, 1, 2, 3, 4, 5])
    struct.add_constraint(4, [0, 1, 2, 3, 4, 5])
    struct.add_force(2, 0, 100e3)

    return _run_analysis(struct, n_mass_matrices, _make_impulse(2.0e-3),
                         2, 0, 0.02, 500,
                         "Exemplo 04 - Pórtico Espacial (Paz)",
                         "Deslocamento do nó 2 (m)", "ex04", plot)


def example_06_truss_3d(n_mass_matrices=6, plot=True):
    print("=" * 70)
    print("EXEMPLO 06 - Treliça tridimensional")
    print("=" * 70)

    E = 2.0e11; G = 0.8e11; rho = 7850.0
    A = 20.0e-4; Iy = 2.667e-6; Iz = 2.667e-6
    J = 4.5e-6; Ix = 5.334e-6; L = 3.0

    struct = Structure(dim='3d', elem_type='truss')
    struct.add_node(1, 0.0, 0.0, 0.0)
    struct.add_node(2, L, 0.0, 0.0)
    struct.add_node(3, L, L, 0.0)
    struct.add_node(4, 0.0, L, 0.0)
    struct.add_node(5, L/2, L/2, L)
    struct.add_element(1, 1, 5, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_element(2, 2, 5, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_element(3, 3, 5, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_element(4, 4, 5, E, A, rho, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)
    struct.add_constraint(1, [0, 1, 2])
    struct.add_constraint(2, [0, 1, 2])
    struct.add_constraint(3, [0, 1, 2])
    struct.add_constraint(4, [0, 1, 2])
    struct.add_force(5, 2, 50e3)

    return _run_analysis(struct, n_mass_matrices, _make_impulse(1.5e-3),
                         5, 2, 0.015, 400,
                         "Exemplo 06 - Treliça 3D",
                         "Deslocamento do nó 5 (m)", "ex06", plot)


def example_cantilever_beam(n_mass_matrices=3, plot=True):
    print("=" * 70)
    print("VALIDAÇÃO - Viga em balanço")
    print("=" * 70)

    E = 210e9; rho = 7850.0
    b = 0.1; h = 0.2
    A = b * h; I = b * h**3 / 12
    L_total = 5.0; n_elem = 10

    struct = Structure(dim='2d', elem_type='frame')
    dx = L_total / n_elem
    for i in range(n_elem + 1):
        struct.add_node(i, i * dx, 0.0)
    for i in range(n_elem):
        struct.add_element(i, i, i + 1, E, A, rho, I=I)
    struct.add_constraint(0, [0, 1, 2])

    beta_L = [1.8751, 4.6941, 7.8548]
    coeff = np.sqrt(E * I / (rho * A * L_total**4))
    freq_analiticas = [(bl**2) * coeff for bl in beta_L]

    print(f"\n  Frequências analíticas (rad/s):")
    for i, f in enumerate(freq_analiticas):
        print(f"    Modo {i+1}: {f:.2f}")

    n_mm_max = min(n_mass_matrices, 2)
    for n_mm in range(1, n_mm_max + 1):
        matrices = struct.compute_mass_matrices(n_mm)
        K0 = matrices[0]
        Mlist = matrices[1:]

        freqs, Phi = solve_eigenvalue_augmented(K0, Mlist)
        print(f"\n  Com {n_mm} MM {'(clássico)' if n_mm == 1 else '(avançado)'}:")
        for i in range(min(3, len(freqs))):
            erro = abs(freqs[i] - freq_analiticas[i]) / freq_analiticas[i] * 100
            print(f"    Modo {i+1}: {freqs[i]:.2f} rad/s (erro: {erro:.2f}%)")


# ============================================================================
# Plotagem
# ============================================================================

def _plot_responses(results, title, ylabel, prefix):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for (label, data), color in zip(results.items(), colors):
        t_arr, d_arr, _ = data
        ax.plot(t_arr, d_arr, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Tempo (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
    plt.tight_layout()
    plt.savefig(f'{prefix}_resposta.png', dpi=150)
    print(f"  Gráfico salvo: {prefix}_resposta.png")
    plt.close()


def _plot_frequencies(results, title, prefix):
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, data in results.items():
        _, _, freqs = data
        x = np.arange(1, len(freqs) + 1)
        ax.plot(x, freqs, 'o-', label=label, markersize=4, linewidth=1.2)
    ax.set_xlabel('Modo', fontsize=12)
    ax.set_ylabel('Frequência (rad/s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{prefix}_frequencias.png', dpi=150)
    print(f"  Gráfico salvo: {prefix}_frequencias.png")
    plt.close()


def run_all_examples(n_mass_matrices=6):
    print("\n" + "=" * 70)
    print("  ANÁLISE DINÂMICA DE TRELIÇAS E PÓRTICOS")
    print("  Técnica de Superposição Modal Avançada")
    print("  Baseado na dissertação de Rodrigo N. Barros (PUC-Rio, 2017)")
    print("=" * 70)

    print("\n\n--- Validação: Viga em balanço ---")
    example_cantilever_beam(n_mass_matrices=3, plot=False)

    print("\n\n--- Exemplo 01: Treliça plana (Weaver) ---")
    example_01_truss_2d(n_mass_matrices, plot=True)

    print("\n\n--- Exemplo 03: Pórtico 2D (Weaver) ---")
    example_03_frame_2d(n_mass_matrices, plot=True)

    print("\n\n--- Exemplo 04: Pórtico espacial (Paz) ---")
    example_04_frame_3d(n_mass_matrices, plot=True)

    print("\n\n--- Exemplo 06: Treliça 3D ---")
    example_06_truss_3d(n_mass_matrices, plot=True)
