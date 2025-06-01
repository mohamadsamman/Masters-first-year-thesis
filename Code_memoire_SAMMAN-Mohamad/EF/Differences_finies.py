
import numpy as np

# Construction du vecteur F_a, de la matrice M et de son inverse
def construire(N, h, k, p, a):

    #discretisation
    x = np.linspace(0, 1, N+2)

    #vecteur contenant les valeur de raideur au centre des cellules
    K_demi = k((x[:-1] + x[1:]) / 2)
    
    #Construction de F_a
    F_a = p(x[1:-1] - a)
    
    #Construction de la matrice M
    diagonale = 1/h**2 * (K_demi[:-1] + K_demi[1:])
    diagonale_inf_sup = -1/h**2 * K_demi[1:-1]
    M = np.diag(diagonale) + np.diag(diagonale_inf_sup, k=1) + np.diag(diagonale_inf_sup, k=-1)

    #Construction de l'inverse
    M_inv = np.linalg.inv(M)
    return  F_a, M, M_inv


#Algo de resolution du systeme lineaire
def resoudre(M_inv,F_a):
    U_a_temp = M_inv@F_a
    U_a = np.zeros(len(U_a_temp))
    U_a = U_a + U_a_temp
    return U_a


# Calcul de l'énergie de rupture R_N(a) et de sa dérivée
def calcul_R_N_et_derivée(a, N, h, k, p, dp_da,M_inv):
    x = np.linspace(0, 1, N+2)    
    F_a = p(x[1:-1] - a)
    U_a = M_inv@F_a  
      
    # Calcul de R_N(a)
    R_N_a = 0.5 * (k(h/2) * U_a[0]/h)**2 + 0.5 * (k(1 - h/2) * U_a[-1]/h)**2
    
    # Préparation pour calculer la dérivée de R_N(a)
    G_a = -np.array([dp_da(xi - a) for xi in x[1:-1]])
    V_a =  M_inv@G_a
    
    # Calcul de la dérivée de R_N(a)
    R_N_prime_a = (k(h/2) * k(h/2) * U_a[0] * V_a[0] / h**2) + (k(1 - h/2) * k(1 - h/2) * U_a[-1] * V_a[-1] / h**2)
    
    return R_N_a, R_N_prime_a

# Méthode de gradient à pas constant
def gradient_pas_constant(N, h, k, p, dp_da, M_inv , rho=0.04, a_init=0.45, tol=1e-6, max_iter=100):
    a = a_init
    iter = 0
    R_N_a, R_N_prime_a = calcul_R_N_et_derivée(a, N, h, k, p, dp_da,M_inv)
    while (abs(R_N_prime_a) > tol and iter < max_iter):
        a = a - rho * R_N_prime_a
        R_N_a, R_N_prime_a = calcul_R_N_et_derivée(a, N, h, k, p, dp_da,M_inv)
        iter = iter + 1
    print(f"Convergence atteinte après {iter} itérations.")
    return a, R_N_a