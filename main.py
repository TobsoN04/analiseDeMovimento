"""
Análise Dinâmica de Treliças e Pórticos Tridimensionais
Usando uma Técnica Avançada de Superposição Modal

Baseado na dissertação de mestrado:
    Rodrigo Nascimento Barros
    PUC-Rio, Maio de 2017
    Orientador: Prof. Ney Augusto Dumont
    Co-orientador: Dr. Carlos Andrés Aguilar Marín

Implementação em Python do programa originalmente desenvolvido em MAPLE CLASSIC.

Módulos:
    elements.py  - Matrizes de rigidez efetiva (Método Híbrido dos Elementos Finitos)
    assembly.py  - Montagem global e condições de contorno
    solver.py    - Autovalores não-lineares e superposição modal avançada
    examples.py  - Exemplos numéricos da dissertação

Uso:
    python main.py                    -> Executa todos os exemplos
    python main.py --example 1        -> Executa exemplo específico (1, 3, 4 ou 6)
    python main.py --nmm 3            -> Usa até 3 matrizes de massa
    python main.py --custom            -> Modo interativo para estrutura personalizada
"""

import sys
import argparse
import numpy as np

from assembly import Structure
from solver import solve_eigenvalue_augmented, normalize_modes, modal_superposition_fast
from examples import (
    example_01_truss_2d,
    example_03_frame_2d,
    example_04_frame_3d,
    example_06_truss_3d,
    example_cantilever_beam,
    run_all_examples,
    heaviside
)


def custom_analysis():
    """Modo interativo para definir uma estrutura personalizada."""
    print("\n" + "=" * 70)
    print("  MODO PERSONALIZADO - Defina sua estrutura")
    print("=" * 70)

    print("\nTipo de estrutura:")
    print("  1 - Treliça 2D")
    print("  2 - Pórtico 2D")
    print("  3 - Treliça 3D")
    print("  4 - Pórtico 3D")

    choice = input("\nEscolha (1-4): ").strip()
    type_map = {
        '1': ('2d', 'truss'),
        '2': ('2d', 'frame'),
        '3': ('3d', 'truss'),
        '4': ('3d', 'frame')
    }

    if choice not in type_map:
        print("Opção inválida.")
        return

    dim, elem_type = type_map[choice]
    struct = Structure(dim=dim, elem_type=elem_type)

    # Propriedades do material
    print("\n--- Propriedades do material ---")
    E = float(input("  Módulo de elasticidade E (Pa) [207e9]: ") or 207e9)
    rho = float(input("  Densidade rho (kg/m³) [7850]: ") or 7850)

    G = None
    if dim == '3d':
        nu = float(input("  Coeficiente de Poisson nu [0.3]: ") or 0.3)
        G = E / (2 * (1 + nu))

    # Nós
    print("\n--- Definição dos nós ---")
    n_nodes = int(input("  Número de nós: "))
    for i in range(n_nodes):
        if dim == '2d':
            coords = input(f"  Nó {i+1} (x, y): ").split(',')
            struct.add_node(i + 1, float(coords[0]), float(coords[1]))
        else:
            coords = input(f"  Nó {i+1} (x, y, z): ").split(',')
            struct.add_node(i + 1, float(coords[0]), float(coords[1]), float(coords[2]))

    # Elementos
    print("\n--- Definição dos elementos ---")
    n_elems = int(input("  Número de elementos: "))
    for i in range(n_elems):
        conn = input(f"  Elemento {i+1} (nó_i, nó_j): ").split(',')
        ni, nj = int(conn[0]), int(conn[1])
        A = float(input(f"    Área A (m²): "))

        I = Iy = Iz = J = Ix = None
        if elem_type == 'frame' or (elem_type == 'truss' and dim == '2d'):
            I = float(input(f"    Momento de inércia I (m⁴): "))
        if dim == '3d':
            Iy = float(input(f"    Iy (m⁴): "))
            Iz = float(input(f"    Iz (m⁴): "))
            J = float(input(f"    J - momento polar (m⁴): "))
            Ix = float(input(f"    Ix (m⁴): "))

        struct.add_element(i + 1, ni, nj, E, A, rho, I=I, Iy=Iy, Iz=Iz, G=G, J=J, Ix=Ix)

    # Restrições
    print("\n--- Condições de contorno ---")
    n_constr = int(input("  Número de nós com restrição: "))
    for _ in range(n_constr):
        node_id = int(input("  Nó restringido: "))
        dofs = input(f"    GDL restringidos (ex: 0,1,2): ").split(',')
        struct.add_constraint(node_id, [int(d) for d in dofs])

    # Carregamento
    print("\n--- Carregamento ---")
    force_node = int(input("  Nó de aplicação da força: "))
    force_dof = int(input("  GDL da força (0=x, 1=y, ...): "))
    force_val = float(input("  Amplitude da força (N): "))
    struct.add_force(force_node, force_dof, force_val)

    t1 = float(input("  Tempo de subida do impulso t1 (s): "))

    def time_func(t):
        t2 = 2 * t1
        return ((t / t1) * heaviside(t) * heaviside(t1 - t)
                + (2 - t / t1) * heaviside(t - t1) * heaviside(t2 - t))

    # Análise
    n_mm = int(input("\n  Número de matrizes de massa (1-6) [6]: ") or 6)
    t_final = float(input("  Tempo final de análise (s) [0.02]: ") or 0.02)
    n_steps = int(input("  Número de passos de tempo [500]: ") or 500)

    print("\n  Executando análise...")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for mm in range(1, n_mm + 1):
        print(f"    {mm} matriz(es) de massa...")
        matrices = struct.compute_global_mass_matrices(mm)
        K0 = matrices[0]
        Mlist = matrices[1:]

        freqs, Phi = solve_eigenvalue_augmented(K0, Mlist)
        if len(freqs) == 0:
            continue

        Phi_norm = normalize_modes(Phi, Mlist[0])
        f_vec = struct.get_force_vector()
        t_array = np.linspace(0, t_final, n_steps)
        d_hist = modal_superposition_fast(freqs, Phi_norm, f_vec, time_func, t_array)

        dof_idx = struct.get_node_dof_in_reduced(force_node, force_dof)
        if dof_idx is not None:
            ax1.plot(t_array, d_hist[dof_idx, :], label=f'{mm}MM')
            ax2.plot(range(1, len(freqs)+1), freqs, 'o-', label=f'{mm}MM', markersize=3)

        print(f"      Frequências: {freqs[:3]}")

    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Deslocamento (m)')
    ax1.set_title('Resposta no Tempo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Modo')
    ax2.set_ylabel('Frequência (rad/s)')
    ax2.set_title('Frequências Naturais')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_analysis.png', dpi=150)
    print(f"\n  Gráfico salvo: custom_analysis.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Análise Dinâmica - Superposição Modal Avançada (Barros, 2017)'
    )
    parser.add_argument('--example', type=int, choices=[1, 3, 4, 6],
                        help='Número do exemplo (1, 3, 4 ou 6)')
    parser.add_argument('--nmm', type=int, default=6,
                        help='Número de matrizes de massa (padrão: 6)')
    parser.add_argument('--custom', action='store_true',
                        help='Modo interativo para estrutura personalizada')
    parser.add_argument('--validate', action='store_true',
                        help='Executa validação com viga em balanço')
    parser.add_argument('--all', action='store_true',
                        help='Executa todos os exemplos')

    args = parser.parse_args()

    if args.custom:
        custom_analysis()
    elif args.validate:
        example_cantilever_beam(n_mass_matrices=args.nmm)
    elif args.example == 1:
        example_01_truss_2d(args.nmm)
    elif args.example == 3:
        example_03_frame_2d(args.nmm)
    elif args.example == 4:
        example_04_frame_3d(args.nmm)
    elif args.example == 6:
        example_06_truss_3d(args.nmm)
    elif args.all:
        run_all_examples(args.nmm)
    else:
        run_all_examples(args.nmm)


if __name__ == '__main__':
    main()
