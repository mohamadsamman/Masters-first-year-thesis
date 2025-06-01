import numpy as np

def nb_ddl_local(degre_EF):
    # Nombre de degres de liberte par element
    # EF de degre 'degre_EF'
    nb_ddl = degre_EF + 1
    return nb_ddl

def FonctionBase(x, degre_EF):
    # Valeur 'phi' (nb_ddl_local(degre_EF) x 1)
    #  des fonctions de base au point de coordonnee 'x' 
    #  pour un element de degre 'degre_EF'
    nb_ddl_loc = nb_ddl_local(degre_EF)
    phi = np.zeros(nb_ddl_loc)
    
    if degre_EF == 0:
        phi[0] = 1.
    elif degre_EF == 1:
        phi[0] = 1. - x
        phi[1] = x
    elif degre_EF == 2:
        phi[0] = 2. * (x - 1.) * (x - 0.5)
        phi[1] = 4. * x * (1. - x)
        phi[2] = 2. * x * (x - 0.5)
    else:
        raise ValueError(f"Les fonctions de base 1D de degre {degre_EF} ne sont pas implementees.")
    
    return phi

def GradientFonctionBase(x, degre_EF):
    # Valeur 'gradphi' (nb_ddl_local(degre_EF) x 1)
    #  de la derivee des fonctions de base au point de coordonnee 'x' 
    #  pour un element de degre 'degre_EF'
    nb_ddl_loc = nb_ddl_local(degre_EF)
    gradphi = np.zeros(nb_ddl_loc)
    
    if degre_EF == 0:
        gradphi[0] = 0.
    elif degre_EF == 1:
        gradphi[0] = -1.
        gradphi[1] = 1.
    elif degre_EF == 2:
        gradphi[0] = 2. * (x - 1.) + 2. * (x - 0.5)
        gradphi[1] = -4. * x + 4. * (1. - x)
        gradphi[2] = 2. * x + 2. * (x - 0.5)
    else:
        raise ValueError(f"Les gradients des fonctions de base 1D de degre {degre_EF} ne sont pas implementes.")    
    return gradphi