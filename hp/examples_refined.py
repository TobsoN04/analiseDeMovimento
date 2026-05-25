"""
Versões refinadas (mais nós) dos exemplos para medir gradiente de discretização.
"""
from mpmath import mp, mpf, pi as mp_pi, sin as mp_sin
from hp.core import Structure, Node


def ex01_truss_weaver_TQ_refined(subdiv=2):
    """
    Treliça Weaver com cada barra dividida em 'subdiv' segmentos.
    Cada segmento ainda é truss 2D.
    """
    E = mpf('207e9'); rho = mpf('7850'); L = mpf('6.35')
    h = L * mp_sin(mp_pi / 3)
    A1, I1 = mpf('64.93e-4'), mpf('16960e-8')
    A2, I2 = mpf('39.12e-4'), mpf('10303e-8')
    A3, I3 = mpf('51.58e-4'), mpf('13530e-8')

    s = Structure(dim='2d', elem_type='truss')

    def add_subdivided_bar(s, n_start_id, p_start, p_end, E, A, rho, I, n_sub, next_id):
        """Adiciona uma barra subdividida em n_sub segmentos."""
        ids = [n_start_id]
        for k in range(1, n_sub):
            t = mpf(k) / mpf(n_sub)
            x = p_start[0] + t * (p_end[0] - p_start[0])
            y = p_start[1] + t * (p_end[1] - p_start[1])
            s.add_node(next_id, x, y)
            ids.append(next_id)
            next_id += 1
        return ids, next_id

    # Nós fixos
    s.add_node(1, 0, 0)
    s.add_node(2, L, 0)
    s.add_node(3, L/2, h)
    next_id = 100

    # Barra 1 (nó 1 → nó 2)
    bar1, next_id = add_subdivided_bar(s, 1, (mpf(0), mpf(0)), (L, mpf(0)), E, A1, rho, I1, subdiv, next_id)
    bar1.append(2)
    elem_id = 1
    for k in range(len(bar1) - 1):
        s.add_element(elem_id, bar1[k], bar1[k+1], E, A1, rho, I=I1)
        elem_id += 1

    # Barra 2 (nó 2 → nó 3)
    bar2, next_id = add_subdivided_bar(s, 2, (L, mpf(0)), (L/2, h), E, A2, rho, I2, subdiv, next_id)
    bar2.append(3)
    for k in range(len(bar2) - 1):
        s.add_element(elem_id, bar2[k], bar2[k+1], E, A2, rho, I=I2)
        elem_id += 1

    # Barra 3 (nó 1 → nó 3)
    bar3, next_id = add_subdivided_bar(s, 1, (mpf(0), mpf(0)), (L/2, h), E, A3, rho, I3, subdiv, next_id)
    bar3.append(3)
    for k in range(len(bar3) - 1):
        s.add_element(elem_id, bar3[k], bar3[k+1], E, A3, rho, I=I3)
        elem_id += 1

    s.add_constraint(1, [1])
    s.add_constraint(3, [0, 1])
    return s


def ex04_frame_paz_refined(subdiv=2):
    """Pórtico 3D Paz com cada barra dividida em 'subdiv' segmentos."""
    from hp.examples import G_GRAVITY
    E = mpf('207e9'); G = mpf('83e9'); L = mpf('5.08')
    A1 = mpf('3.23e-2'); I1 = mpf('8.32e-5'); J1 = mpf('1.66e-5')
    A2 = mpf('1.81e-2'); I2 = mpf('2.66e-5'); J2 = mpf('5.33e-6')
    m1 = mpf('140.62'); m2 = mpf('70.31')
    rho1 = m1 * G_GRAVITY / A1
    rho2 = m2 * G_GRAVITY / A2

    s = Structure(dim='3d', elem_type='frame')

    next_id = 100
    s.add_node(1, L, 0, 0)
    s.add_node(2, 0, 0, 0)
    s.add_node(3, 0, L, 0)
    s.add_node(4, 0, 0, L)

    def add_bar(start_id, end_id, props, e_start):
        ids = [start_id]
        n_i = s.nodes[start_id]; n_j = s.nodes[end_id]
        for k in range(1, subdiv):
            t = mpf(k) / mpf(subdiv)
            x = n_i.x + t*(n_j.x - n_i.x)
            y = n_i.y + t*(n_j.y - n_i.y)
            z = n_i.z + t*(n_j.z - n_i.z)
            nonlocal_id = next_id_ref[0]
            s.add_node(nonlocal_id, x, y, z)
            ids.append(nonlocal_id)
            next_id_ref[0] += 1
        ids.append(end_id)
        for k in range(len(ids) - 1):
            s.add_element(e_start + k, ids[k], ids[k+1],
                           props['E'], props['A'], props['rho'],
                           Iy=props['Iy'], Iz=props['Iz'], G=props['G'],
                           J=props['J'], Ix=props['Iy'])
        return e_start + (len(ids) - 1)

    next_id_ref = [next_id]
    p1 = {'E':E,'G':G,'A':A1,'rho':rho1,'Iy':I1,'Iz':I1,'J':J1}
    p2 = {'E':E,'G':G,'A':A2,'rho':rho2,'Iy':I2,'Iz':I2,'J':J2}
    e_id = 1
    e_id = add_bar(2, 1, p1, e_id) + 1
    e_id = add_bar(2, 3, p2, e_id) + 1
    e_id = add_bar(2, 4, p1, e_id) + 1

    s.add_constraint(1, [0, 1, 2, 3, 4, 5])
    s.add_constraint(3, [0, 1, 2, 3, 4, 5])
    s.add_constraint(4, [0, 1, 2, 3, 4, 5])
    return s


def ex06_truss3d_paz_refined(subdiv=2):
    """Treliça 3D Paz com cada barra dividida em 'subdiv' segmentos."""
    E = mpf('207e9'); A = mpf('6.452e-3')
    m_per_L = mpf('670'); rho = m_per_L / A
    G = mpf('80e9'); L = mpf('4.0')
    I = mpf('15753e-8'); J = 2*I

    s = Structure(dim='3d', elem_type='truss')
    s.add_node(1, 0, 0, 0)
    s.add_node(2, L, 0, 0)
    s.add_node(3, L, L, 0)
    s.add_node(4, 0, L, 0)
    s.add_node(5, L/2, L/2, L)

    next_id = [100]
    e_id = 1
    for ni_id in [1, 2, 3, 4]:
        n_i = s.nodes[ni_id]; n_j = s.nodes[5]
        ids = [ni_id]
        for k in range(1, subdiv):
            t = mpf(k) / mpf(subdiv)
            x = n_i.x + t*(n_j.x - n_i.x)
            y = n_i.y + t*(n_j.y - n_i.y)
            z = n_i.z + t*(n_j.z - n_i.z)
            s.add_node(next_id[0], x, y, z)
            ids.append(next_id[0])
            next_id[0] += 1
        ids.append(5)
        for k in range(len(ids) - 1):
            s.add_element(e_id, ids[k], ids[k+1], E, A, rho, Iy=I, Iz=I, G=G, J=J, Ix=J)
            e_id += 1

    for ni in [1, 2, 3, 4]:
        s.add_constraint(ni, [0, 1, 2])
    return s
