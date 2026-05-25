"""
Montagem de matrizes globais e aplicação de condições de contorno.

Monta K0, M1 analiticamente, e extrai M2, M3, ... numericamente usando
K(ω) em frequências bem escolhidas.
"""

import numpy as np
from elements import (
    beam_2d_stiffness, truss_2d_stiffness,
    beam_3d_stiffness, truss_3d_stiffness,
    beam_2d_K0_M1, truss_2d_K0_M1,
    beam_3d_K0_M1, truss_3d_K0_M1,
    beam_2d_M2, truss_2d_M2,
    beam_3d_M2, truss_3d_M2,
    rotation_matrix_2d, rotation_matrix_2d_truss,
    rotation_matrix_3d,
    transform_matrix, build_T_3d_beam, build_T_3d_truss
)


class Node:
    def __init__(self, id, x, y, z=0.0):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


class Element:
    def __init__(self, id, node_i, node_j, E, A, rho,
                 I=None, Iy=None, Iz=None, G=None, J=None, Ix=None):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.A = A
        self.rho = rho
        self.I = I
        self.Iy = Iy
        self.Iz = Iz
        self.G = G
        self.J = J
        self.Ix = Ix

    @property
    def length(self):
        ni = self.node_i
        nj = self.node_j
        return np.sqrt((nj.x-ni.x)**2 + (nj.y-ni.y)**2 + (nj.z-ni.z)**2)


class Structure:
    def __init__(self, dim='2d', elem_type='truss'):
        self.dim = dim
        self.elem_type = elem_type
        self.nodes = {}
        self.elements = []
        self.constraints = {}
        self.forces = {}

        if dim == '2d' and elem_type == 'truss':
            self.dof_per_node = 2
        elif dim == '2d' and elem_type == 'frame':
            self.dof_per_node = 3
        elif dim == '3d' and elem_type == 'truss':
            self.dof_per_node = 3
        elif dim == '3d' and elem_type == 'frame':
            self.dof_per_node = 6

    def add_node(self, id, x, y, z=0.0):
        self.nodes[id] = Node(id, x, y, z)

    def add_element(self, id, ni_id, nj_id, E, A, rho,
                    I=None, Iy=None, Iz=None, G=None, J=None, Ix=None):
        ni = self.nodes[ni_id]
        nj = self.nodes[nj_id]
        self.elements.append(Element(id, ni, nj, E, A, rho, I, Iy, Iz, G, J, Ix))

    def add_constraint(self, node_id, dofs):
        self.constraints[node_id] = dofs

    def add_force(self, node_id, dof, value):
        if node_id not in self.forces:
            self.forces[node_id] = {}
        self.forces[node_id][dof] = value

    def _global_dofs(self, node_id):
        sorted_ids = sorted(self.nodes.keys())
        pos = sorted_ids.index(node_id)
        start = pos * self.dof_per_node
        return list(range(start, start + self.dof_per_node))

    def _get_free_dofs(self):
        total = len(self.nodes) * self.dof_per_node
        constrained = set()
        for node_id, dofs in self.constraints.items():
            gdofs = self._global_dofs(node_id)
            for d in dofs:
                constrained.add(gdofs[d])
        return [i for i in range(total) if i not in constrained]

    def _get_element_transform(self, elem):
        """Retorna a matriz de transformação T e os GDL globais do elemento."""
        ni = elem.node_i
        nj = elem.node_j
        dofs_i = self._global_dofs(ni.id)
        dofs_j = self._global_dofs(nj.id)
        elem_dofs = dofs_i + dofs_j

        if self.dim == '2d' and self.elem_type == 'truss':
            T = rotation_matrix_2d_truss(ni.x, ni.y, nj.x, nj.y)
        elif self.dim == '2d' and self.elem_type == 'frame':
            T = rotation_matrix_2d(ni.x, ni.y, nj.x, nj.y)
        elif self.dim == '3d' and self.elem_type == 'truss':
            R = rotation_matrix_3d(ni.x, ni.y, ni.z, nj.x, nj.y, nj.z)
            T = build_T_3d_truss(R)
        elif self.dim == '3d' and self.elem_type == 'frame':
            R = rotation_matrix_3d(ni.x, ni.y, ni.z, nj.x, nj.y, nj.z)
            T = build_T_3d_beam(R)

        return T, elem_dofs

    def _element_K0_M1_local(self, elem):
        """K0 e M1 analíticos no sistema local."""
        L = elem.length
        if self.dim == '2d' and self.elem_type == 'truss':
            return truss_2d_K0_M1(elem.E, elem.A, elem.I, elem.rho, L)
        elif self.dim == '2d' and self.elem_type == 'frame':
            return beam_2d_K0_M1(elem.E, elem.A, elem.I, elem.rho, L)
        elif self.dim == '3d' and self.elem_type == 'truss':
            return truss_3d_K0_M1(elem.E, elem.G, elem.A,
                                   elem.Iy, elem.Iz, elem.J, elem.Ix, elem.rho, L)
        elif self.dim == '3d' and self.elem_type == 'frame':
            return beam_3d_K0_M1(elem.E, elem.G, elem.A,
                                  elem.Iy, elem.Iz, elem.J, elem.Ix, elem.rho, L)

    def _element_K_omega_local(self, elem, omega):
        """K(ω) no sistema local."""
        L = elem.length
        if self.dim == '2d' and self.elem_type == 'truss':
            return truss_2d_stiffness(elem.E, elem.A, elem.I, elem.rho, L, omega)
        elif self.dim == '2d' and self.elem_type == 'frame':
            return beam_2d_stiffness(elem.E, elem.A, elem.I, elem.rho, L, omega)
        elif self.dim == '3d' and self.elem_type == 'truss':
            return truss_3d_stiffness(elem.E, elem.G, elem.A,
                                       elem.Iy, elem.Iz, elem.J, elem.Ix, elem.rho, L, omega)
        elif self.dim == '3d' and self.elem_type == 'frame':
            return beam_3d_stiffness(elem.E, elem.G, elem.A,
                                      elem.Iy, elem.Iz, elem.J, elem.Ix, elem.rho, L, omega)

    def _assemble(self, local_matrices_func):
        """Monta uma matriz global a partir de matrizes locais transformadas."""
        total = len(self.nodes) * self.dof_per_node
        G = np.zeros((total, total))
        for elem in self.elements:
            K_local = local_matrices_func(elem)
            T, elem_dofs = self._get_element_transform(elem)
            K_global = transform_matrix(K_local, T)
            for i_l, i_g in enumerate(elem_dofs):
                for j_l, j_g in enumerate(elem_dofs):
                    G[i_g, j_g] += K_global[i_l, j_l]
        return G

    def _reduce(self, M_full):
        """Reduz uma matriz global para GDL livres."""
        free = self._get_free_dofs()
        return M_full[np.ix_(free, free)]

    def assemble_K0_M1(self):
        """Monta K0 e M1 globais reduzidos (analíticos)."""
        total = len(self.nodes) * self.dof_per_node
        K0_full = np.zeros((total, total))
        M1_full = np.zeros((total, total))

        for elem in self.elements:
            K0_loc, M1_loc = self._element_K0_M1_local(elem)
            T, elem_dofs = self._get_element_transform(elem)
            K0_glob = transform_matrix(K0_loc, T)
            M1_glob = transform_matrix(M1_loc, T)
            for i_l, i_g in enumerate(elem_dofs):
                for j_l, j_g in enumerate(elem_dofs):
                    K0_full[i_g, j_g] += K0_glob[i_l, j_l]
                    M1_full[i_g, j_g] += M1_glob[i_l, j_l]

        free = self._get_free_dofs()
        return K0_full[np.ix_(free, free)], M1_full[np.ix_(free, free)]

    def assemble_K_omega(self, omega):
        """Monta K(ω) global reduzido usando expressões fechadas."""
        total = len(self.nodes) * self.dof_per_node
        K_full = np.zeros((total, total))

        for elem in self.elements:
            K_loc = self._element_K_omega_local(elem, omega)
            T, elem_dofs = self._get_element_transform(elem)
            K_glob = transform_matrix(K_loc, T)
            for i_l, i_g in enumerate(elem_dofs):
                for j_l, j_g in enumerate(elem_dofs):
                    K_full[i_g, j_g] += K_glob[i_l, j_l]

        free = self._get_free_dofs()
        return K_full[np.ix_(free, free)]

    def _element_M2_local(self, elem):
        """M2 analítica no sistema local."""
        L = elem.length
        if self.dim == '2d' and self.elem_type == 'truss':
            return truss_2d_M2(elem.E, elem.A, elem.I, elem.rho, L)
        elif self.dim == '2d' and self.elem_type == 'frame':
            return beam_2d_M2(elem.E, elem.A, elem.I, elem.rho, L)
        elif self.dim == '3d' and self.elem_type == 'truss':
            return truss_3d_M2(elem.E, elem.G, elem.A,
                                elem.Iy, elem.Iz, elem.J, elem.Ix, elem.rho, L)
        elif self.dim == '3d' and self.elem_type == 'frame':
            return beam_3d_M2(elem.E, elem.G, elem.A,
                               elem.Iy, elem.Iz, elem.J, elem.Ix, elem.rho, L)

    def assemble_M2(self):
        """Monta M2 global reduzida (analítica)."""
        total = len(self.nodes) * self.dof_per_node
        M2_full = np.zeros((total, total))
        for elem in self.elements:
            M2_loc = self._element_M2_local(elem)
            T, elem_dofs = self._get_element_transform(elem)
            M2_glob = transform_matrix(M2_loc, T)
            for i_l, i_g in enumerate(elem_dofs):
                for j_l, j_g in enumerate(elem_dofs):
                    M2_full[i_g, j_g] += M2_glob[i_l, j_l]
        free = self._get_free_dofs()
        return M2_full[np.ix_(free, free)]

    def compute_mass_matrices(self, n_matrices):
        """Retorna [K0, M1] ou [K0, M1, M2] conforme n_matrices."""
        K0, M1 = self.assemble_K0_M1()
        if n_matrices <= 1:
            return [K0, M1]
        M2 = self.assemble_M2()
        return [K0, M1, M2]

    def assemble_K_omega_nmm(self, omega, n_mm):
        return self.assemble_K_omega(omega)

    def get_force_vector(self):
        total = len(self.nodes) * self.dof_per_node
        f = np.zeros(total)
        for node_id, dof_vals in self.forces.items():
            gdofs = self._global_dofs(node_id)
            for dof, val in dof_vals.items():
                f[gdofs[dof]] = val
        free = self._get_free_dofs()
        return f[free]

    def get_node_dof_in_reduced(self, node_id, local_dof):
        gdofs = self._global_dofs(node_id)
        target = gdofs[local_dof]
        free = self._get_free_dofs()
        return free.index(target) if target in free else None
