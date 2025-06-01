

import numpy as np
import matplotlib.pyplot as plt
import time
import Differences_finies as df

# Paramètres

N = 101
h = 1. / (N + 1)
a = 0.4
lbda = 20.  # λ dans la fonction p
cst1 = 1.

#differentes raideurs

#k = lambda x: cst1 + 0*x
#k = lambda x: 1. - (x-0.5)*(x-0.5)
#k = lambda x: 1. + (x-0.5)*(x-0.5)
#k = lambda x:0.5*np.sin(12*(x-0.1))+1
#k = lambda x:np.abs(1e6 * 0.5*np.sin(12*(x-0.1))+1) #ATTENTION PAS C1
#k = lambda x: np.exp(-12*(x-0.5)**2)
k = lambda x: 1. + 0.8 * np.sin(10*x) * np.cos(5*x)


p = lambda x: -lbda * np.exp(-(lbda * x)**2)
dp_da = lambda x: 2 * lbda**3 * (x) * np.exp(-lbda**2 * (x)**2)


# Exécution de l'algorithme de minimisation
F_a,M,M_inv=df.construire(N,h,k,p,a)
tic=time.perf_counter()
a_opt, R_N_opt = df.gradient_pas_constant( N , h, k, p, dp_da,M_inv,rho=0.04 ,a_init=a,tol=1e-6,max_iter=500)
toc=time.perf_counter()
print(f"calcul effectué en {toc-tic} secondes")
print(f"a_opt: {a_opt}, R_N_opt: {R_N_opt}")



# Affichage de la solution approchée
F_a,M,M_inv=df.construire(N,h,k,p,a_opt)
U_a = df.resoudre(M_inv,F_a)
x = np.linspace(0, 1, len(U_a))
plt.figure(1)
plt.plot(x, U_a, label=f'Solution approchée pour a = {a_opt}')
plt.xlabel('Position x')
plt.ylabel('Déplacement u_a(x)')
plt.legend()

plt.figure(2)
K_x = k(x)
plt.plot(x, K_x, label=f'la raideur')
plt.xlabel('Position x')
plt.ylabel('La tension k(x)')
plt.legend()

plt.figure(3)
F_a = p(x[1:-1] - a_opt)
plt.plot(x[1:-1], F_a , label=f'la charge optimale')
plt.xlabel('Position x')
plt.ylabel('la charge f(x)')
plt.legend()

plt.show()

#Courbe de Rn en fonction de a
aa=np.linspace(0.1,0.9,100)
Rn=np.zeros_like(aa)
for i in range (len(aa)): 
    F_a,M,M_inv=df.construire(N,h,k,p,aa[i]) 
    Rn[i],temp=df.calcul_R_N_et_derivée(aa[i], N, h, k, p, dp_da,M_inv)
plt.plot(aa,Rn)
plt.show()  
