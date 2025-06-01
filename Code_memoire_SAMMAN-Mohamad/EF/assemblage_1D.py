import numpy as np
import formules_integration_1D as fi
import elements_de_reference_1D as elt



#-----------------------------------------------------------------#
def ErreurL2( uexact, uu, degre_EF, degre_quadrature, Sommets, nb_cells, Cellules, NumGlob ):
    """
      Erreur L2 entre une valeur exacte et une solution approchee
      uexact : fonction exacte
      uu (nb_ddl_global(degre_EF, nb_cells) x 1) : valeur des degres de liberte globaux definissant l'approximation
      degre_EF : degre de l'element de reference
      degre_quadrature : degre de la règle de quadrature
      Sommets (nb_sommtx1) : coordonnees de chaque sommet
      nb_cells : nombre de cellules
      Cellules (nb_cellsx2) : numero du sommets de chaque cellule
      NumGlob (nb_ddl_local(degre_EF) x nb_cells) : Numerotation globale des degres de liberte locaux
    """

    nb_ddl_loc = elt.nb_ddl_local(degre_EF)

    omega, xi, npoints = fi.Quadrature1d(degre_quadrature)  # 2*degre_EF+2

    err = 0.
    for ic in range(nb_cells):
        ac = Sommets[Cellules[ic, 0]]
        bc = Sommets[Cellules[ic, 1]]

        uu_loc = np.zeros(nb_ddl_loc)
        for il in range(nb_ddl_loc):
            uu_loc[il] = uu[NumGlob[il, ic]]

        err_loc = 0.
        for k in range(npoints):
            phi = elt.FonctionBase(xi[k], degre_EF)
            err_k = uexact((bc - ac) * xi[k] + ac) - np.sum(uu_loc * phi)
            err_loc += (bc - ac) * omega[k] * err_k**2

        err += err_loc

    err = np.sqrt(err)
    return err
#-----------------------------------------------------------------#
def nb_ddl_global(degre_EF, nb_cells):
    """
    Nombre total de degres de liberte
    EF de degre 'degre_EF' sur un maillage 1D avec 'nb_cells' cellules
    """
    nb_ddl = degre_EF * nb_cells + 1
    return nb_ddl
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def NumerotationGlobale(degre_EF, nb_cells):
    """
    Numerotation globale des ddl
    EF de degre 'degre_EF' sur un maillage 1D avec 'nb_cells' cellules
    """
    nb_ddl_loc = elt.nb_ddl_local(degre_EF)
    NumGlob = np.zeros((nb_ddl_loc, nb_cells),dtype=int)
    for ic in range(nb_cells):
        for il in range(nb_ddl_loc):
            NumGlob[il, ic] = ic * (nb_ddl_loc - 1) + il
    NumGlobBord = np.array( [0, nb_cells * (nb_ddl_loc - 1)] )
    return NumGlob, NumGlobBord
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def add_row(f, ic, degre_quadrature, Cellules, Sommets, degre_EF):
    nb_ddl_loc = elt.nb_ddl_local(degre_EF)
    ac = Sommets[Cellules[ic, 0]]
    bc = Sommets[Cellules[ic, 1]]
    omega, xi, npoints = fi.Quadrature1d(degre_quadrature)
    vec_loc = np.zeros(nb_ddl_loc)
    for k in range(npoints):
        Phi = elt.FonctionBase(xi[k], degre_EF)
        vec_loc += (bc - ac) * omega[k] * f( (bc - ac) * (xi[k]) + ac ) * Phi
    return vec_loc
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def add_row_col(ic, degre_quadrature, Cellules, Sommets, degre_EF):
    nb_ddl_loc = elt.nb_ddl_local(degre_EF)
    ac = Sommets[Cellules[ic, 0]]
    bc = Sommets[Cellules[ic, 1]]
    omega, xi, npoints = fi.Quadrature1d(degre_quadrature)
    mat_loc = np.zeros((nb_ddl_loc, nb_ddl_loc))
    for k in range(npoints):
        phi = elt.FonctionBase(xi[k], degre_EF)
        mat_loc += (bc - ac) * omega[k] * np.outer(phi,phi)
    return mat_loc
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def add_grad_row_grad_col(ic, degre_quadrature, Cellules, Sommets, degre_EF,K):
    nb_ddl_loc = elt.nb_ddl_local(degre_EF)
    ac = Sommets[Cellules[ic, 0]]
    bc = Sommets[Cellules[ic, 1]]
    omega, xi, npoints = fi.Quadrature1d(degre_quadrature)
    mat_loc = np.zeros((nb_ddl_loc, nb_ddl_loc))
    for k in range(npoints):
        Gphi = 1 / (bc - ac) * elt.GradientFonctionBase(xi[k], degre_EF)
        mat_loc += (bc - ac) * omega[k]*K((bc - ac) * xi[k] + ac ) *np.outer(Gphi,Gphi)
    return mat_loc
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def assemble_dans_vecteur(vec_glob, ic, vec_loc, NumGlob):
    rvec_glob = vec_glob.copy()
    for il in range(NumGlob.shape[0]):
        ig = NumGlob[il, ic]
        rvec_glob[ig] += vec_loc[il]
    return rvec_glob
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def assemble_dans_matrice(mat_glob, ic, mat_loc, NumGlob):
    rmat_glob = mat_glob.copy()
    for il in range(NumGlob.shape[0]):
        ig = NumGlob[il, ic]
        for jl in range(NumGlob.shape[0]):
            jg = NumGlob[jl, ic]
            rmat_glob[ig, jg] += mat_loc[il, jl]
    return rmat_glob
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def Masse(degre_EF, Sommets, nb_cells, Cellules, NumGlob):
    nb_ddl_glob = nb_ddl_global(degre_EF, nb_cells)
    degre_quadrature = 2 * degre_EF
    M = np.zeros((nb_ddl_glob, nb_ddl_glob))
    for ic in range(nb_cells):
        mat_loc = add_row_col(ic, degre_quadrature, Cellules, Sommets, degre_EF)
        M = assemble_dans_matrice(M, ic, mat_loc, NumGlob)
    return M
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def ProjectionSurEspaceEF(f, degre_EF, Sommets, nb_cells, Cellules, NumGlob):
    nb_ddl_glob = nb_ddl_global(degre_EF, nb_cells)
    degre_quadrature = 2 * degre_EF + 3
    inte = np.zeros(nb_ddl_glob)
    for ic in range(nb_cells):
        vec_loc = add_row(f, ic, degre_quadrature, Cellules, Sommets, degre_EF)
        inte = assemble_dans_vecteur(inte, ic, vec_loc, NumGlob)
    return inte
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
def Rigidite(degre_EF, Sommets, nb_cells, Cellules, NumGlob,K):
    nb_ddl_glob = nb_ddl_global(degre_EF, nb_cells)
    degre_quadrature = 2 * (degre_EF - 1)

    R = np.zeros((nb_ddl_glob, nb_ddl_glob))

    for ic in range(nb_cells):
        mat_loc = add_grad_row_grad_col(ic, degre_quadrature, Cellules, Sommets, degre_EF,K)
        R = assemble_dans_matrice(R, ic, mat_loc, NumGlob)

    return R
#-----------------------------------------------------------------#
