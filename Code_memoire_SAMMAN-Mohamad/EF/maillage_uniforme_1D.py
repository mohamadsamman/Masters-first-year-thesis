import numpy as np

def UniformMesh1d(a, b, nb_cells):
    """
    Crée un maillage 1D uniforme du segment ['a', 'b'] contenant 'nb_cells' cellules.
    nb_sommt : nombre de sommets
    Sommets (nb_sommtx1) : coordonnés de chaque sommet
    nb_cells : nombre de cellules
    Cellules (nb_cellsx2) : numero du sommets de chaque cellule
    """
    
    nb_cells = nb_cells
    nb_sommt = nb_cells + 1
    nb_aretebord = 2

    h = (b-a) / nb_cells
    
    # Creation des sommets
    # Sommets = np.zeros(nb_sommt)
    # for i in np.arange(nb_sommt)
    #    Sommets[i] = a + i*h
    # end
    Sommets = a*np.ones(nb_sommt) + h*np.arange(nb_sommt)
    
    # Creation des cellules
    # Cellules = np.zeros((nb_cells,2))
    # for i in np.arange(nb_cells)
    #    Cellules[i,1] = i
    #    Cellules[i,2] = i+1
    # end
    Cellules = np.zeros((nb_cells,2),dtype=int)
    Cellules[:,0] = np.arange(nb_cells)
    Cellules[:,1] = np.arange(1,nb_cells+1)

    return Sommets, nb_sommt, Cellules, nb_cells

