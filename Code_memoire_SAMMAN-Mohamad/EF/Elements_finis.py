
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import maillage_uniforme_1D as msh
import assemblage_1D as asmb
import time

def ef(a,b,N,degre_EF,f,K,A):
    Sommets, nb_sommt, Cellules, nb_cells = msh.UniformMesh1d(a, b, N)
    NumGlob, NumGlobBord = asmb.NumerotationGlobale(degre_EF, nb_cells)
    
    rhs = asmb.ProjectionSurEspaceEF(f, degre_EF, Sommets, nb_cells, Cellules, NumGlob)
    R = asmb.Rigidite(degre_EF, Sommets, nb_cells, Cellules, NumGlob,K)
    
    # Conditions aux bords
    iddl_bg = NumGlobBord[0]
    R[iddl_bg, :] = 0
    R[iddl_bg, iddl_bg] = 1
    rhs[iddl_bg] = 0
    
    iddl_bd = NumGlobBord[1]
    R[iddl_bd, :] = 0
    R[iddl_bd, iddl_bd] = 1
    rhs[iddl_bd] = 0
    
    uh = la.solve(R, rhs)
    return uh

# Calcul de l'énergie de rupture R_N(a) et de sa dérivée
def calcul_R_N_et_derivée_ef(a,b,A, N, h, K, p, dp_da,degre_EF):
    x = np.linspace(0, 1, N+1)    
    U_a=ef(a,b,N,degre_EF,p,K,A)
    V_a=ef(a,b,N,degre_EF,dp_da,K,A)  
    
    # Calcul de R_N(a)
    R_N_a = 0.5 * (K(h/2) * U_a[1]/h)**2 + 0.5 * (K(1 - h/2) * U_a[-2]/h)**2

    
    # Calcul de la dérivée de R_N(a)
    R_N_prime_a = (K(h/2) * K(h/2) * U_a[1] * V_a[1] / h**2) + (K(1 - h/2) * K(1 - h/2) * U_a[-2] * V_a[-2] / h**2)
    
    return R_N_a, R_N_prime_a,U_a

def gradient_pas_constant_ef(a,b, N, h, k, p, dp_da, degre_EF , rho=0.04, a_init=0.45, tol=1e-6, max_iter=110):
    A = a_init
    iter = 0
    R_N_a, R_N_prime_a,U_a = calcul_R_N_et_derivée_ef(a,b,A, N, h, K, p, dp_da,degre_EF)
    print(R_N_a,R_N_prime_a)
    while (abs(R_N_prime_a) > tol and iter < max_iter):
        A = A - rho * R_N_prime_a
        p = lambda x: -lbda * np.exp(-(lbda * (x-A))**2)
        dp_da = lambda x: -2 * lbda**3 * (x-A) * np.exp(-lbda**2 * (x-A)**2)
        R_N_a, R_N_prime_a,U_a = calcul_R_N_et_derivée_ef(a,b,A, N, h, K, p, dp_da,degre_EF)
        iter = iter + 1
    print(f"Convergence atteinte après {iter} itérations.")
    return A, R_N_a,U_a

#Paramètres
N = 100
h=1./(N+1)
A = 0.5
degre_EF=1
lbda = 20.  # λ dans la fonction p
cst1=1.
#differentes raideurs

#K = lambda x: cst1 + 0*x
#K = lambda x: 1. - (x-0.5)*(x-0.5)
#K = lambda x: 1. + (x-0.5)*(x-0.5)
#K = lambda x:0.5*np.sin(12*(x-0.1))+1
#K = lambda x:np.abs(1e6 * 0.5*np.sin(12*(x-0.1))+1) #ATTENTION PAS C1
#K = lambda x: np.exp(-12*(x-0.5)**2)
K = lambda x: 1. + 0.8 * np.sin(10*x) * np.cos(5*x)



p = lambda x: -lbda * np.exp(-(lbda * (x-A))**2)
dp_da = lambda x: -2 * lbda**3 * (x-A) * np.exp(-lbda**2 * (x-A)**2)



#calcul de la solution approchée
tic=time.perf_counter()
a_opt, R_N_opt,U_a = gradient_pas_constant_ef( 0,1, N, h, K, p, dp_da, degre_EF , rho=0.04, a_init=A, tol=1e-6, max_iter=400)
toc=time.perf_counter()
print(f"calcul effectué en {toc-tic} secondes")
print(f"a_opt: {a_opt}, R_N_opt: {R_N_opt}")

#affichage de la solution
x=np.linspace(0,1,len(U_a))
plt.plot(x, U_a, label=f'Solution approchée pour a = {a_opt}')
plt.xlabel('Position x')
plt.ylabel('Déplacement u_a(x)')
plt.legend()
plt.show()

#erreur relatif entre differences finies et elements finis pour k(x) = 1. + 0.8 * np.sin(10*x) * np.cos(5*x) 
delta_a = np.abs((0.666678380283545-a_opt))/ 0.666678380283545
delta_R = np.abs((0.7853981633974548-R_N_opt))/ 0.7853981633974548
print("l'erreur relative |DF-EF/DF| de la position du centre de la charge  est: ",delta_a)
print("l'erreur relative |DF-EF/DF| du risque de rupture est: ",delta_R)

#norme sup DF-EF
import Differences_finies as df

F_a,M,M_inv=df.construire(N+1,h,K,p,a_opt)
U_a_df = df.resoudre(M_inv,F_a)

diff=np.max(np.abs(U_a_df-U_a))
print(f"norme sup: {diff}")


#moindres carres
mc=0
for i in range(len(U_a)):
    mc=mc+(U_a[i]-U_a_df[i])**2
print(f"moindre carres: {mc}")



