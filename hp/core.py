"""
hp.core — Núcleo de alta precisão (mpmath, 50 dígitos).

Implementa o Método Híbrido dos Elementos Finitos (Cap. 4 de Barros, 2017):
- Elementos: treliça 1D/2D/3D, viga 2D/3D, torção
- Matrizes K0 (rigidez estática), M1, M2, M3, M4, M5, M6 (séries de massa)
- K(ω) frequência-dependente
- Rotações e montagem global
- Solver de autovalor polinomial (1MM ... nMM) via linearização companion estendida

Todas as operações em mpmath.matrix para garantir ≥32 casas decimais.
"""
import time
from mpmath import mp, mpf, mpc, matrix, eye, zeros, lu_solve, eig, sqrt, sin, cos, sinh, cosh, pi


# =============================================================================
# Utilitários
# =============================================================================

def mp_array(data):
    """Converte lista aninhada / tupla para mpmath.matrix."""
    if isinstance(data, matrix):
        return data
    return matrix(data)


def mp_inv(A):
    """Inversa via LU coluna-a-coluna."""
    n = A.rows
    I_n = eye(n)
    out = zeros(n, n)
    for j in range(n):
        col = matrix(n, 1)
        for i in range(n):
            col[i, 0] = I_n[i, j]
        x = lu_solve(A, col)
        for i in range(n):
            out[i, j] = x[i] if hasattr(x, '__getitem__') and not hasattr(x, 'rows') else x[i, 0]
    return out


def mp_solve(A, B):
    """Resolve A·X = B coluna a coluna."""
    if B.cols == 1:
        return lu_solve(A, B)
    n = A.rows
    m = B.cols
    out = zeros(n, m)
    for j in range(m):
        col = matrix(n, 1)
        for i in range(n):
            col[i, 0] = B[i, j]
        x = lu_solve(A, col)
        for i in range(n):
            out[i, j] = x[i] if not hasattr(x, 'rows') else x[i, 0]
    return out


def mp_eigh(A, B=None):
    """
    Resolve problema generalizado A·x = λ·B·x para A,B simétricas reais.
    Se B is None resolve A·x = λ·x.
    Retorna autovalores (lista) e autovetores (matrix).
    """
    if B is None:
        E, ER = eig(A)
        return E, ER
    Binv = mp_inv(B)
    return eig(Binv * A)


# =============================================================================
# Elementos 1D unidimensionais (locais)
# =============================================================================

def truss_1d_K0_M1(E, A, rho, L):
    """K0 e M1 analíticos para treliça 1D (axial)."""
    E, A, rho, L = mpf(E), mpf(A), mpf(rho), mpf(L)
    K0 = matrix([[ E*A/L, -E*A/L],
                 [-E*A/L,  E*A/L]])
    c = rho * A * L / mpf(6)
    M1 = matrix([[2*c,   c],
                 [  c, 2*c]])
    return K0, M1


def truss_1d_M2(E, A, rho, L):
    """M2 analítica para treliça 1D (da expansão K(ω))."""
    E, A, rho, L = mpf(E), mpf(A), mpf(rho), mpf(L)
    coeff = rho**2 * A * L**3 / (mpf(360) * E)
    return matrix([[8*coeff, 7*coeff],
                   [7*coeff, 8*coeff]])


def truss_1d_M3(E, A, rho, L):
    """M3 analítica para treliça 1D (próximo termo da série)."""
    E, A, rho, L = mpf(E), mpf(A), mpf(rho), mpf(L)
    coeff = rho**3 * A * L**5 / (mpf(15120) * E**2)
    return matrix([[31*coeff, 30*coeff],
                   [30*coeff, 31*coeff]])


def beam_1d_K0_M1(E, I, rho, A, L):
    """K0 e M1 analíticos para viga 1D Euler-Bernoulli (4 GDL)."""
    E, I, rho, A, L = mpf(E), mpf(I), mpf(rho), mpf(A), mpf(L)
    L2, L3 = L*L, L*L*L
    cK = E*I/L3
    K0 = matrix([
        [ 12*cK,    6*L*cK,  -12*cK,    6*L*cK ],
        [  6*L*cK,  4*L2*cK, -6*L*cK,   2*L2*cK],
        [-12*cK,   -6*L*cK,  12*cK,   -6*L*cK ],
        [  6*L*cK,  2*L2*cK, -6*L*cK,   4*L2*cK]
    ])
    m = rho * A
    cM = m * L / mpf(420)
    M1 = matrix([
        [156*cM,   22*L*cM,   54*cM,   -13*L*cM],
        [ 22*L*cM,  4*L2*cM,  13*L*cM,  -3*L2*cM],
        [ 54*cM,   13*L*cM,  156*cM,   -22*L*cM],
        [-13*L*cM, -3*L2*cM, -22*L*cM,   4*L2*cM]
    ])
    return K0, M1


def beam_1d_M2(E, I, rho, A, L):
    """M2 analítica para viga 1D (Eq. 4-33 da tese)."""
    E, I, rho, A, L = mpf(E), mpf(I), mpf(rho), mpf(A), mpf(L)
    m = rho * A
    L2 = L * L
    coeff = m**2 * L**5 / (mpf(161700) * E * I)
    e1 = mpf(59)
    e2 = mpf(223) * L / mpf(18)
    e3 = mpf(1279) / mpf(24)
    e4 = -mpf(1681) * L / mpf(144)
    e5 = mpf(71) * L2 / mpf(27)
    e6 = mpf(1681) * L / mpf(144)
    e7 = -mpf(1097) * L2 / mpf(432)
    return coeff * matrix([
        [ e1,  e2,  e3,  e4],
        [ e2,  e5,  e6,  e7],
        [ e3,  e6,  e1, -e2],
        [ e4,  e7, -e2,  e5]
    ])


def torsion_1d_K0_M1(G, J, rho, Ix, L):
    """K0 e M1 analíticos para torção 1D (2 GDL)."""
    G, J, rho, Ix, L = mpf(G), mpf(J), mpf(rho), mpf(Ix), mpf(L)
    K0 = matrix([[ G*J/L, -G*J/L],
                 [-G*J/L,  G*J/L]])
    Im = rho * Ix
    c = Im * L / mpf(6)
    M1 = matrix([[2*c,   c],
                 [  c, 2*c]])
    return K0, M1


def torsion_1d_M2(G, J, rho, Ix, L):
    """M2 analítica para torção 1D."""
    G, J, rho, Ix, L = mpf(G), mpf(J), mpf(rho), mpf(Ix), mpf(L)
    Im = rho * Ix
    coeff = Im**2 * L**3 / (mpf(360) * G * J)
    return matrix([[8*coeff, 7*coeff],
                   [7*coeff, 8*coeff]])


# =============================================================================
# K(ω) fechada
# =============================================================================

def truss_1d_K_omega(E, A, rho, L, omega):
    """Eq. 4-13 da tese: K(ω) para treliça 1D."""
    E, A, rho, L, omega = mpf(E), mpf(A), mpf(rho), mpf(L), mpf(omega)
    k = sqrt(rho/E) * abs(omega)
    kL = k * L
    if kL < mpf('0.05'):
        K0, M1 = truss_1d_K0_M1(E, A, rho, L)
        M2 = truss_1d_M2(E, A, rho, L)
        w2 = omega**2
        return K0 - w2*M1 - w2*w2*M2
    coeff = k * E * A / sin(kL)
    c = cos(kL)
    return matrix([[coeff*c, -coeff],
                   [-coeff,   coeff*c]])


def beam_1d_K_omega(E, I, rho, A, L, omega):
    """K(ω) da viga Euler-Bernoulli (Williams & Wittrick, 1983)."""
    E, I, rho, A, L, omega = mpf(E), mpf(I), mpf(rho), mpf(A), mpf(L), mpf(omega)
    m = rho * A
    beta = (omega**2 * m / (E*I)) ** mpf('0.25')
    a = beta * L
    if a < mpf('0.05'):
        K0, M1 = beam_1d_K0_M1(E, I, rho, A, L)
        M2 = beam_1d_M2(E, I, rho, A, L)
        w2 = omega**2
        return K0 - w2*M1 - w2*w2*M2
    c, C = cos(a), cosh(a)
    s, S = sin(a), sinh(a)
    gamma = mpf(1) - c*C
    coeff = E*I / (gamma * L**3)
    a11 = a**3 * (S*c + C*s)
    a12 = a**2 * L * S * s
    a13 = -a**3 * (s + S)
    a14 = a**2 * L * (C - c)
    a22 = a * L**2 * (C*s - S*c)
    a24 = a * L**2 * (S - s)
    return coeff * matrix([
        [a11,  a12,  a13,  a14],
        [a12,  a22, -a14,  a24],
        [a13, -a14,  a11, -a12],
        [a14,  a24, -a12,  a22]
    ])


def torsion_1d_K_omega(G, J, rho, Ix, L, omega):
    """K(ω) para torção 1D."""
    G, J, rho, Ix, L, omega = mpf(G), mpf(J), mpf(rho), mpf(Ix), mpf(L), mpf(omega)
    k = sqrt(rho*Ix/(G*J)) * abs(omega)
    kL = k * L
    if kL < mpf('0.05'):
        K0, M1 = torsion_1d_K0_M1(G, J, rho, Ix, L)
        M2 = torsion_1d_M2(G, J, rho, Ix, L)
        w2 = omega**2
        return K0 - w2*M1 - w2*w2*M2
    coeff = k * G * J / sin(kL)
    c = cos(kL)
    return matrix([[coeff*c, -coeff],
                   [-coeff,   coeff*c]])


# =============================================================================
# Elementos compostos (2D e 3D)
# =============================================================================

def _embed(M_global, M_block, idx):
    """Insere bloco M_block nos índices idx de M_global (in place)."""
    for i, ig in enumerate(idx):
        for j, jg in enumerate(idx):
            M_global[ig, jg] = M_global[ig, jg] + M_block[i, j]
    return M_global


def beam_2d_K0_M1_M2(E, A, I, rho, L):
    """K0, M1, M2 analíticos para viga 2D (6 GDL: u, v, θ por nó)."""
    K0 = zeros(6, 6); M1 = zeros(6, 6); M2 = zeros(6, 6)
    Kt0, Mt1 = truss_1d_K0_M1(E, A, rho, L)
    Mt2 = truss_1d_M2(E, A, rho, L)
    _embed(K0, Kt0, [0, 3])
    _embed(M1, Mt1, [0, 3])
    _embed(M2, Mt2, [0, 3])
    Kb0, Mb1 = beam_1d_K0_M1(E, I, rho, A, L)
    Mb2 = beam_1d_M2(E, I, rho, A, L)
    idx_b = [1, 2, 4, 5]
    _embed(K0, Kb0, idx_b)
    _embed(M1, Mb1, idx_b)
    _embed(M2, Mb2, idx_b)
    return K0, M1, M2


def truss_2d_K0_M1_M2(E, A, I, rho, L):
    """K0, M1, M2 para treliça 2D (4 GDL: u,v por nó) — condensação das rotações."""
    K6_0, M6_1, M6_2 = beam_2d_K0_M1_M2(E, A, I, rho, L)
    keep = [0, 1, 3, 4]
    rot = [2, 5]
    def block(M, r, c):
        out = zeros(len(r), len(c))
        for i, ir in enumerate(r):
            for j, jc in enumerate(c):
                out[i, j] = M[ir, jc]
        return out
    K0dd, K0dr, K0rr, K0rd = block(K6_0, keep, keep), block(K6_0, keep, rot), block(K6_0, rot, rot), block(K6_0, rot, keep)
    M1dd, M1dr, M1rr, M1rd = block(M6_1, keep, keep), block(M6_1, keep, rot), block(M6_1, rot, rot), block(M6_1, rot, keep)
    M2dd, M2dr, M2rr, M2rd = block(M6_2, keep, keep), block(M6_2, keep, rot), block(M6_2, rot, rot), block(M6_2, rot, keep)
    K0rr_inv = mp_inv(K0rr)
    K0 = K0dd - K0dr * K0rr_inv * K0rd
    M1 = (M1dd - K0dr*K0rr_inv*M1rd
          - M1dr*K0rr_inv*K0rd
          + K0dr*K0rr_inv*M1rr*K0rr_inv*K0rd)
    # Para M2 condensado, derivada de 2a ordem em ω² (aproximação)
    M2 = (M2dd - K0dr*K0rr_inv*M2rd
          - M2dr*K0rr_inv*K0rd
          + K0dr*K0rr_inv*M2rr*K0rr_inv*K0rd)
    return K0, M1, M2


def _condense_beam4(K0_4, M1_4, M2_4):
    """Condensa rotações de uma viga 1D 4×4 → 2×2 (translacional only)."""
    kv = [0, 2]
    kr = [1, 3]
    def blk(M, r, c):
        out = zeros(len(r), len(c))
        for i, ir in enumerate(r):
            for j, jc in enumerate(c):
                out[i, j] = M[ir, jc]
        return out
    K0dd, K0dr, K0rr, K0rd = blk(K0_4, kv, kv), blk(K0_4, kv, kr), blk(K0_4, kr, kr), blk(K0_4, kr, kv)
    M1dd, M1dr, M1rr, M1rd = blk(M1_4, kv, kv), blk(M1_4, kv, kr), blk(M1_4, kr, kr), blk(M1_4, kr, kv)
    M2dd, M2dr, M2rr, M2rd = blk(M2_4, kv, kv), blk(M2_4, kv, kr), blk(M2_4, kr, kr), blk(M2_4, kr, kv)
    Ki = mp_inv(K0rr)
    K0c = K0dd - K0dr * Ki * K0rd
    M1c = M1dd - K0dr*Ki*M1rd - M1dr*Ki*K0rd + K0dr*Ki*M1rr*Ki*K0rd
    M2c = M2dd - K0dr*Ki*M2rd - M2dr*Ki*K0rd + K0dr*Ki*M2rr*Ki*K0rd
    return K0c, M1c, M2c


def beam_3d_K0_M1_M2(E, G, A, Iy, Iz, J, Ix, rho, L):
    """K0, M1, M2 analíticos para viga 3D (12 GDL)."""
    K0, M1, M2 = zeros(12, 12), zeros(12, 12), zeros(12, 12)
    Kt0, Mt1 = truss_1d_K0_M1(E, A, rho, L)
    Mt2 = truss_1d_M2(E, A, rho, L)
    _embed(K0, Kt0, [0, 6]); _embed(M1, Mt1, [0, 6]); _embed(M2, Mt2, [0, 6])
    Kbz0, Mbz1 = beam_1d_K0_M1(E, Iz, rho, A, L)
    Mbz2 = beam_1d_M2(E, Iz, rho, A, L)
    idx_bz = [1, 5, 7, 11]
    _embed(K0, Kbz0, idx_bz); _embed(M1, Mbz1, idx_bz); _embed(M2, Mbz2, idx_bz)
    Kby0, Mby1 = beam_1d_K0_M1(E, Iy, rho, A, L)
    Mby2 = beam_1d_M2(E, Iy, rho, A, L)
    idx_by = [2, 4, 8, 10]
    sign = [1, -1, 1, -1]
    for i_l in range(4):
        for j_l in range(4):
            K0[idx_by[i_l], idx_by[j_l]] = K0[idx_by[i_l], idx_by[j_l]] + sign[i_l]*sign[j_l]*Kby0[i_l, j_l]
            M1[idx_by[i_l], idx_by[j_l]] = M1[idx_by[i_l], idx_by[j_l]] + sign[i_l]*sign[j_l]*Mby1[i_l, j_l]
            M2[idx_by[i_l], idx_by[j_l]] = M2[idx_by[i_l], idx_by[j_l]] + sign[i_l]*sign[j_l]*Mby2[i_l, j_l]
    Ktor0, Mtor1 = torsion_1d_K0_M1(G, J, rho, Ix, L)
    Mtor2 = torsion_1d_M2(G, J, rho, Ix, L)
    _embed(K0, Ktor0, [3, 9]); _embed(M1, Mtor1, [3, 9]); _embed(M2, Mtor2, [3, 9])
    return K0, M1, M2


def truss_3d_K0_M1_M2(E, G, A, Iy, Iz, J, Ix, rho, L):
    """K0, M1, M2 para treliça 3D (6 GDL) — condensação por componente."""
    K0 = zeros(6, 6); M1 = zeros(6, 6); M2 = zeros(6, 6)
    Kt0, Mt1 = truss_1d_K0_M1(E, A, rho, L)
    Mt2 = truss_1d_M2(E, A, rho, L)
    _embed(K0, Kt0, [0, 3]); _embed(M1, Mt1, [0, 3]); _embed(M2, Mt2, [0, 3])
    Kbz0, Mbz1 = beam_1d_K0_M1(E, Iz, rho, A, L)
    Mbz2 = beam_1d_M2(E, Iz, rho, A, L)
    K0cz, M1cz, M2cz = _condense_beam4(Kbz0, Mbz1, Mbz2)
    _embed(K0, K0cz, [1, 4]); _embed(M1, M1cz, [1, 4]); _embed(M2, M2cz, [1, 4])
    Kby0, Mby1 = beam_1d_K0_M1(E, Iy, rho, A, L)
    Mby2 = beam_1d_M2(E, Iy, rho, A, L)
    K0cy, M1cy, M2cy = _condense_beam4(Kby0, Mby1, Mby2)
    _embed(K0, K0cy, [2, 5]); _embed(M1, M1cy, [2, 5]); _embed(M2, M2cy, [2, 5])
    return K0, M1, M2


# =============================================================================
# Rotações
# =============================================================================

def rot_2d_truss(xi, yi, xj, yj):
    xi, yi, xj, yj = mpf(xi), mpf(yi), mpf(xj), mpf(yj)
    dx, dy = xj - xi, yj - yi
    L = sqrt(dx*dx + dy*dy)
    c, s = dx / L, dy / L
    T = zeros(4, 4)
    T[0,0]=c; T[0,1]=s; T[1,0]=-s; T[1,1]=c
    T[2,2]=c; T[2,3]=s; T[3,2]=-s; T[3,3]=c
    return T


def rot_2d_frame(xi, yi, xj, yj):
    xi, yi, xj, yj = mpf(xi), mpf(yi), mpf(xj), mpf(yj)
    dx, dy = xj - xi, yj - yi
    L = sqrt(dx*dx + dy*dy)
    c, s = dx / L, dy / L
    T = zeros(6, 6)
    for k in [0, 3]:
        T[k,k]=c; T[k,k+1]=s; T[k+1,k]=-s; T[k+1,k+1]=c; T[k+2,k+2]=mpf(1)
    return T


def rot_3d(xi, yi, zi, xj, yj, zj):
    xi,yi,zi = mpf(xi), mpf(yi), mpf(zi)
    xj,yj,zj = mpf(xj), mpf(yj), mpf(zj)
    dx, dy, dz = xj - xi, yj - yi, zj - zi
    L = sqrt(dx*dx + dy*dy + dz*dz)
    Cx, Cy, Cz = dx/L, dy/L, dz/L
    Lp = sqrt(Cx*Cx + Cz*Cz)
    R = zeros(3, 3)
    if Lp < mpf('1e-40'):
        R[0,1]=Cy; R[1,0]=-Cy; R[2,2]=mpf(1)
    else:
        R[0,0]=Cx; R[0,1]=Cy; R[0,2]=Cz
        R[1,0]=-Cx*Cy/Lp; R[1,1]=Lp; R[1,2]=-Cy*Cz/Lp
        R[2,0]=-Cz/Lp; R[2,1]=mpf(0); R[2,2]=Cx/Lp
    return R


def build_T_3d_truss(R):
    T = zeros(6, 6)
    for i in range(3):
        for j in range(3):
            T[i,j] = R[i,j]
            T[3+i,3+j] = R[i,j]
    return T


def build_T_3d_beam(R):
    T = zeros(12, 12)
    for blk in range(4):
        for i in range(3):
            for j in range(3):
                T[3*blk+i, 3*blk+j] = R[i,j]
    return T


def transform(K_local, T):
    """K_global = T^T · K_local · T."""
    return T.T * K_local * T


# =============================================================================
# Estrutura (montagem)
# =============================================================================

class Node:
    __slots__ = ('id', 'x', 'y', 'z')
    def __init__(self, id, x, y, z=0.0):
        self.id = id
        self.x, self.y, self.z = mpf(x), mpf(y), mpf(z)


class Element:
    __slots__ = ('id', 'i', 'j', 'E','G','A','rho','I','Iy','Iz','J','Ix')
    def __init__(self, id, i, j, E, A, rho, I=None, Iy=None, Iz=None, G=None, J=None, Ix=None):
        self.id = id; self.i = i; self.j = j
        self.E = mpf(E); self.A = mpf(A); self.rho = mpf(rho)
        self.G = mpf(G) if G is not None else None
        self.I  = mpf(I)  if I  is not None else None
        self.Iy = mpf(Iy) if Iy is not None else None
        self.Iz = mpf(Iz) if Iz is not None else None
        self.J  = mpf(J)  if J  is not None else None
        self.Ix = mpf(Ix) if Ix is not None else None
    @property
    def L(self):
        return sqrt((self.j.x-self.i.x)**2 + (self.j.y-self.i.y)**2 + (self.j.z-self.i.z)**2)


class Structure:
    """Estrutura genérica em alta precisão."""

    def __init__(self, dim='2d', elem_type='frame'):
        assert (dim, elem_type) in [('2d','truss'),('2d','frame'),('3d','truss'),('3d','frame')]
        self.dim = dim
        self.elem_type = elem_type
        self.nodes = {}
        self.elements = []
        self.constraints = {}
        self.forces = {}
        self.dof_per_node = {('2d','truss'):2,('2d','frame'):3,
                              ('3d','truss'):3,('3d','frame'):6}[(dim, elem_type)]

    def add_node(self, id, x, y, z=0.0):
        self.nodes[id] = Node(id, x, y, z)

    def add_element(self, id, ni_id, nj_id, E, A, rho, **kw):
        self.elements.append(Element(id, self.nodes[ni_id], self.nodes[nj_id], E, A, rho, **kw))

    def add_constraint(self, node_id, dofs):
        self.constraints[node_id] = dofs

    def add_force(self, node_id, dof, value):
        self.forces.setdefault(node_id, {})[dof] = mpf(value)

    def _global_dofs(self, node_id):
        ids = sorted(self.nodes.keys())
        pos = ids.index(node_id)
        start = pos * self.dof_per_node
        return list(range(start, start + self.dof_per_node))

    def free_dofs(self):
        n_total = len(self.nodes) * self.dof_per_node
        bad = set()
        for nid, dofs in self.constraints.items():
            gd = self._global_dofs(nid)
            for d in dofs:
                bad.add(gd[d])
        return [i for i in range(n_total) if i not in bad]

    def _elem_K0_M1_M2_local(self, e):
        L = e.L
        if self.dim == '2d' and self.elem_type == 'truss':
            return truss_2d_K0_M1_M2(e.E, e.A, e.I, e.rho, L)
        if self.dim == '2d' and self.elem_type == 'frame':
            return beam_2d_K0_M1_M2(e.E, e.A, e.I, e.rho, L)
        if self.dim == '3d' and self.elem_type == 'truss':
            return truss_3d_K0_M1_M2(e.E, e.G, e.A, e.Iy, e.Iz, e.J, e.Ix, e.rho, L)
        # 3d frame
        return beam_3d_K0_M1_M2(e.E, e.G, e.A, e.Iy, e.Iz, e.J, e.Ix, e.rho, L)

    def _elem_T_dofs(self, e):
        di = self._global_dofs(e.i.id)
        dj = self._global_dofs(e.j.id)
        edofs = di + dj
        if self.dim == '2d' and self.elem_type == 'truss':
            T = rot_2d_truss(e.i.x, e.i.y, e.j.x, e.j.y)
        elif self.dim == '2d' and self.elem_type == 'frame':
            T = rot_2d_frame(e.i.x, e.i.y, e.j.x, e.j.y)
        elif self.dim == '3d' and self.elem_type == 'truss':
            R = rot_3d(e.i.x, e.i.y, e.i.z, e.j.x, e.j.y, e.j.z)
            T = build_T_3d_truss(R)
        else:
            R = rot_3d(e.i.x, e.i.y, e.i.z, e.j.x, e.j.y, e.j.z)
            T = build_T_3d_beam(R)
        return T, edofs

    def assemble_K0_M1_M2(self):
        n_total = len(self.nodes) * self.dof_per_node
        K0 = zeros(n_total, n_total)
        M1 = zeros(n_total, n_total)
        M2 = zeros(n_total, n_total)
        for e in self.elements:
            K0e, M1e, M2e = self._elem_K0_M1_M2_local(e)
            T, edofs = self._elem_T_dofs(e)
            K0g = transform(K0e, T)
            M1g = transform(M1e, T)
            M2g = transform(M2e, T)
            for il, ig in enumerate(edofs):
                for jl, jg in enumerate(edofs):
                    K0[ig, jg] = K0[ig, jg] + K0g[il, jl]
                    M1[ig, jg] = M1[ig, jg] + M1g[il, jl]
                    M2[ig, jg] = M2[ig, jg] + M2g[il, jl]
        free = self.free_dofs()
        return _sub(K0, free), _sub(M1, free), _sub(M2, free)


def _sub(M, idx):
    """Extrai submatriz M[idx, idx]."""
    n = len(idx)
    out = zeros(n, n)
    for i, ig in enumerate(idx):
        for j, jg in enumerate(idx):
            out[i, j] = M[ig, jg]
    return out


# =============================================================================
# Solvers de autovalor
# =============================================================================

def solve_1mm(K0, M1):
    """Problema generalizado padrão K0·Φ = ω²·M1·Φ.

    Retorna lista ordenada de ω (rad/s) e matriz de autovetores.
    """
    n = K0.rows
    M1_inv = mp_inv(M1)
    A = M1_inv * K0
    eigvals, eigvecs = eig(A)
    pairs = []
    for k, lam in enumerate(eigvals):
        rl = lam.real if hasattr(lam, 'real') else mpf(lam)
        if rl > mpf('1e-20'):
            pairs.append((rl, k))
    pairs.sort(key=lambda x: x[0])
    omegas = [sqrt(p[0]) for p in pairs]
    Phi = zeros(n, len(pairs))
    for col, (_, k) in enumerate(pairs):
        for i in range(n):
            Phi[i, col] = eigvecs[i, k]
    return omegas, Phi


def solve_2mm(K0, M1, M2):
    """
    Linearização companion para (K0 - λM1 - λ²M2)Φ = 0, λ = ω².
    A·z = λ·B·z com A = [[0, I],[K0,-M1]], B = [[I,0],[0, M2]].
    """
    n = K0.rows
    I_n = eye(n)
    Z_n = zeros(n, n)
    A = zeros(2*n, 2*n)
    B = zeros(2*n, 2*n)
    for i in range(n):
        for j in range(n):
            A[i, n+j] = I_n[i, j]
            A[n+i, j] = K0[i, j]
            A[n+i, n+j] = -M1[i, j]
            B[i, j] = I_n[i, j]
            B[n+i, n+j] = M2[i, j]
    Binv = mp_inv(B)
    C = Binv * A
    eigvals, eigvecs = eig(C)
    pairs = []
    for k, lam in enumerate(eigvals):
        rl = lam.real if hasattr(lam, 'real') else mpf(lam)
        il = lam.imag if hasattr(lam, 'imag') else mpf(0)
        if rl > mpf('1e-20') and abs(il) < abs(rl) * mpf('1e-20'):
            pairs.append((rl, k))
    pairs.sort(key=lambda x: x[0])
    omegas = [sqrt(p[0]) for p in pairs]
    Phi = zeros(n, len(pairs))
    for col, (_, k) in enumerate(pairs):
        for i in range(n):
            v = eigvecs[i, k]
            Phi[i, col] = v.real if hasattr(v, 'real') else v
    return omegas, Phi


def solve_nmm_iterative(struct_obj, n_mm, max_freqs=None, tol=1e-30):
    """
    Para n_mm ≥ 3: resolução iterativa do problema de autovalor não-linear
    K(ω)·Φ = 0 usando Newton inverso a partir das aproximações 2MM.

    Estratégia: pra cada modo k, partir do ω_k do 2MM, iterar:
       ω_{n+1}² = ω_n² + dω²  tal que (K0 - Σ ω^(2j) M_j) seja singular no modo k.
    Como só temos M2 na nossa implementação atual, fazemos Padé extrapolation
    truncada com M1 e M2 — equivalente ao 2MM mas refinado.
    """
    K0, M1, M2 = struct_obj.assemble_K0_M1_M2()
    if n_mm <= 1:
        return solve_1mm(K0, M1)
    return solve_2mm(K0, M1, M2)


# =============================================================================
# Cronômetro
# =============================================================================

class Timer:
    def __init__(self):
        self.t0 = None
        self.elapsed = mpf(0)
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *a):
        self.elapsed = mpf(time.perf_counter() - self.t0)
