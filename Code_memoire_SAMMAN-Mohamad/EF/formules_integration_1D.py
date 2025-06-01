import numpy as np

def Quadrature1d(degre_quadrature):
    if degre_quadrature in [0, 1]:
        # Gauss, 1 point
        n = 1
        omega = np.zeros(n)
        xi = np.zeros(n)

        xi[0] = 0.5

        omega[0] = 1.0
    elif degre_quadrature in [2, 3]:
        # Gauss, 2 points
        n = 2
        omega = np.zeros(n)
        xi = np.zeros(n)

        omega[0] = 0.5
        omega[1] = 0.5

        xi[0] = 0.21132486540518711775
        xi[1] = 0.78867513459481288225
    elif degre_quadrature in [4, 5]:
        # Gauss, 3 points
        n = 3
        omega = np.zeros(n)
        xi = np.zeros(n)

        omega[0] = 0.277777777778
        omega[1] = 0.444444444444
        omega[2] = 0.277777777778

        xi[0] = 0.11270166537925831148
        xi[1] = 0.5
        xi[2] = 0.88729833462074168852
    elif degre_quadrature in [6, 7]:
        # Gauss, 4 points
        n = 4
        omega = np.zeros(n)
        xi = np.zeros(n)

        xi[0] = 0.069431844203
        xi[1] = 0.330009478208
        xi[2] = 0.669990521792
        xi[3] = 0.930568155797

        omega[0] = 0.173927422569
        omega[1] = 0.326072577431
        omega[2] = 0.326072577431
        omega[3] = 0.173927422569
    elif degre_quadrature in [8, 9]:
        # Gauss, 5 points
        n = 5
        omega = np.zeros(n)
        xi = np.zeros(n)

        xi[0] = 0.0469100770307
        xi[1] = 0.230765344947
        xi[2] = 0.5
        xi[3] = 0.769234655053
        xi[4] = 0.953089922969

        omega[0] = 0.118463442528
        omega[1] = 0.23931433525
        omega[2] = 0.284444444444
        omega[3] = 0.23931433525
        omega[4] = 0.118463442528
    elif degre_quadrature in [10, 11]:
        # Gauss, 6 points
        n = 6
        omega = np.zeros(n)
        xi = np.zeros(n)

        xi[0] = 0.03376524289842398608
        xi[1] = 0.16939530676686774318
        xi[2] = 0.38069040695840154568
        xi[3] = 0.61930959304159845432
        xi[4] = 0.83060469323313225682
        xi[5] = 0.96623475710157601392

        d = 1.0 / np.sqrt(0.0073380204222450993933)
        omega[0] = d * 0.0073380204222450993933
        omega[1] = d * 0.015451823343095832149
        omega[2] = d * 0.020041279329451654676
        omega[3] = d * 0.020041279329451654676
        omega[4] = d * 0.015451823343095832149
        omega[5] = d * 0.0073380204222450993933
    else:
        raise ValueError(f"Aucune formule de quadrature 1d de degre {degre_quadrature} n'est implementee.\n")

    return omega, xi, n

