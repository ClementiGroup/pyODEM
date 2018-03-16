""" Helper functions for help in computing non-bonded and native interactions

Written primarily by Fabio Trovato in Cecilia Clementi's Group.
GitHub: https://github.com/fabiotrovato
With help from Justin Chen (TensorDuck) in Cecilia Clementi's group.

"""

import time
import sys
import mdtraj as md
import numpy as np
import math
from pysph.base.utils import get_particle_array
from pysph.base import utils
from pysph.base import nnps
from pyzoltan.core.carray import UIntArray
import argparse
from argparse import RawTextHelpFormatter

#This is the dictionary of the potentials.
dict_pot = {
    # pot_id : ['pot name', num params]
    1: ['LJ',2], #this is in the defaults section, nbfunc key... is it a 12-10 pot for CG models?
    6: ['LJ12GAUSSIAN',4]
        }
####################### METHODS SPECIFICALLY FOR PyODEM ########################

def convert_c6c12_to_sigma_eps(c6, c12):
    sigma = (( 6. / 5.) * (c12/c6)) ** 0.5
    eps = ((5. / c12) ** 5) * ((c6 / 6.) ** 6)

    return sigma, eps

def convert_sigma_eps_to_c6c12(sigma, eps):
    c6 = 6 * (sigma**10) * eps
    c12 = 5 * (sigma**12) * eps

    return c6, c12

def order_epsilons_atm_types(dict_atm_types, n_types):
    # get an ordered list of eps and sigmas from a dictionary of atom types
    epsilons_atm_types = np.zeros(n_types)
    sigmas_atm_types = np.zeros(n_types)
    for k1, v1 in dict_atm_types.iteritems():
        idx = v1[0]
        c6 = v1[1]
        c12 = v1[2]
        sigma, eps = convert_c6c12_to_sigma_eps(c6, c12)
        assert sigma > 0
        assert eps > 0
        epsilons_atm_types[idx] = eps
        sigmas_atm_types[idx] = sigma

    return epsilons_atm_types, sigmas_atm_types

def compute_mixed_table(params, mix_rule):
    # Compute a mixing of parameters via mix_rules
    n_params = np.shape(params)[0]
    params_mat = np.zeros((n_params, n_params))
    for idx in range(n_params):
        for jdx in range(idx, n_params):
            if mix_rule[0] == 1:
                params_mat[idx, jdx] = math.sqrt(params[idx] * params[jdx])
                params_mat[jdx, idx] = params_mat[idx, jdx]
            else:
                print "Mixing rule not defined in compute_mixed_table()"

    return params_mat

def compute_mixed_derivatives_table(params):
    n_params = np.shape(params)[0]
    params_mat = np.zeros((n_params, n_params))
    for idx in range(n_params):
        for jdx in range(n_params):
            params_mat[idx,jdx] = 0.5*math.sqrt(params[jdx]/params[idx])

def get_c6c12_matrix_noeps(sigmas, mix_rule):
    # compute the C6/eps and C12/eps parameters.
    # return param_mat, which is C6 and C12 for each pair type divided by eps_ij
    n_params = np.shape(sigmas)[0]
    mixed_table = compute_mixed_table(sigmas, mix_rule)
    param_mat = [ [None for i in range(n_params)] for j in range(n_params) ]
    for idx in range(n_params):
        for jdx in range(n_params):
            c6 = 6 * (mixed_table[idx,jdx]**10)
            c12 = 5 * (mixed_table[idx,jdx]**12)
            param_mat[idx][jdx] = [c6,c12]

    return param_mat




#################### METHODS FROM FABIO ########################################
def split_pairwise_atmtype(nl_pairs, pairsidx_ps):
    # nl_pairs: list of pair of indices, those belonging to the neigh list. They are both pairs subject to pots in the [ pairs ] section and [ atomtypes ] section.
    # for each nl_pairs check if it belongs to [ pairs ]. If not, then the pots to be used are those specified in [ atomtypes ].
    #
    # ps = pairs section
    # as = atomtype section
    # return nl_ps, nl_atmtyp that are respectively the pairs in the nl that belongs to [ pairs ] or [ atomtype ] sections

    nl_ps = []
    nl_atmtyp = []


    for p in nl_pairs:

        pair = [p[0]+1, p[1]+1]

        # check if pair belongs to the [ pairs ] section
        if pair in pairsidx_ps:
            nl_ps.append(p)
            #nl_ps.append([p[0], p[1]])
            #np.append(nl_ps, [p[0], p[1]])
        else:
            #np.append(nl_atmtyp, [p[0], p[1]])
            #nl_atmtyp.append([p[0], p[1]])
            nl_atmtyp.append(p)

    return nl_ps, nl_atmtyp

def calc_nb_ene_fast(dist_array, pot_type, parms):
    """ Compute the non-bonded energy for each component

    Assuming you have N CA atom pairs that are close enough to have an
    interaction, the non-bonded potential energy is then computed for those atom
    pairs. The method currently supports gaussian and LJ12-10 type interactions.

    Unlike calc_nb_ene(), does two things differently. First, instead of
    returning the total energy, it returns an array of each pair energy. Second,
    it computes the potential energy without epsilons.

    args:
        dist_array (array floats): An array of length N that contains the
            distances computed for each pair.
        pot_type (list of int): Each entry denotes the GROMACS index for the
            potential type. Each entry i denotes the potential to use for i'th
            potential. Example: 1=Lennard Jones and 6=Gaussian interaction.
        parms (list of list of floats): List of len(N), where each entry i is a
            list of floats to use in the i'th potential energy calculation.

    returns:
        U_list (array of float): An array of len N for each nonbonded energy

    """
    # This function calculates the energy, given a variety of potential functional forms
    # given an array of distances, the pot_type and parms as inputs. printf is a flag
    # If printf = True, prints on file the single interactions.

    # FUNDAMENTAL ASSUMPTIONS:
    # The i-th elements in each list are "correspondent", which implies that
    # all lists have the same size
    # they are ordered, i.e. the i-th parms is appropriate for the i-th distance and pot_type

    # lowercase u means a single potential. Uppercase U means a sum.

    strngdst = ""
    strngpot = ""
    U_list = np.zeros(np.shape(dist_array)[0])
    for k, x in enumerate(dist_array):
        u = 0.0
        if(pot_type[k] == 6): #LJ12GAUSSIAN
            #assumes no mistakes in params array
            eps = parms[k][0]
            x1 = parms[k][1]
            s1 = parms[k][2]
            x0_12 = parms[k][3]

            #params combination
            x_12 =  math.pow(x, 12)
            s1_2 = math.pow(s1,2)
            dx = x-x1
            dx_2 = math.pow(dx, 2)

            # potential evaluation (formula below has been validated by direct comparison with a traj of 100 frames of a 2-particle system)
            rep = (1.0 + (1.0/eps)*(x0_12/x_12))
            g = math.exp((-dx_2)/(2.0*s1_2))
            u = -g

        elif pot_type[k] == 1: #LJ (12-10 for cg models, I think)
            p0 = parms[k][0]
            p1 = parms[k][1]
            #params combination
            x_10 =  math.pow(x, 10)
            #x_10 =  math.pow(x, 6)
            x_12 =  math.pow(x, 12)

            # potential evaluation (formula below has been validated by direct comparison with a traj of 100 frames of a 2-particle system)
            u = - p0/x_10 + p1/x_12


            #print k,x,pot_type[k],p0,p1,u, U

        else:
            print "********************************************"
            print "Code unavailable for this type of potential:",pot_type[k]
            print "********************************************"
            exit()

        U_list[k] = u


    return U_list


def calc_nb_ene(dist_array, pot_type, parms, printf):
    """ Compute the total non-bonded energy

    Assuming you have N CA atom pairs that are close enough to have an
    interaction, the non-bonded potential energy is then computed for those atom
    pairs. The method currently supports gaussian and LJ12-10 type interactions.

    args:
        dist_array (array floats): An array of length N that contains the
            distances computed for each pair.
        pot_type (list of int): Each entry denotes the GROMACS index for the
            potential type. Each entry i denotes the potential to use for i'th
            potential. Example: 1=Lennard Jones and 6=Gaussian interaction.
        parms (list of list of floats): List of len(N), where each entry i is a
            list of floats to use in the i'th potential energy calculation.

    returns:
        U (float): The total non-bonded potential energy.

    """
    # This function calculates the energy, given a variety of potential functional forms
    # given an array of distances, the pot_type and parms as inputs. printf is a flag
    # If printf = True, prints on file the single interactions.

    # FUNDAMENTAL ASSUMPTIONS:
    # The i-th elements in each list are "correspondent", which implies that
    # all lists have the same size
    # they are ordered, i.e. the i-th parms is appropriate for the i-th distance and pot_type

    # lowercase u means a single potential. Uppercase U means a sum.

    strngdst = ""
    strngpot = ""
    U = 0.0

    for k, x in enumerate(dist_array):
        u = 0.0
        if(pot_type[k] == 6): #LJ12GAUSSIAN
            #assumes no mistakes in params array
            eps = parms[k][0]
            x1 = parms[k][1]
            s1 = parms[k][2]
            x0_12 = parms[k][3]

            #params combination
            x_12 =  math.pow(x, 12)
            s1_2 = math.pow(s1,2)
            dx = x-x1
            dx_2 = math.pow(dx, 2)

            # potential evaluation (formula below has been validated by direct comparison with a traj of 100 frames of a 2-particle system)
            rep = (1.0 + (1.0/eps)*(x0_12/x_12))
            g = math.exp((-dx_2)/(2.0*s1_2))
            w = 1.0 - g
            #u = eps*(rep*w - 1.0)
            u = -g # for testing purposes
            U += u


        elif pot_type[k] == 1: #LJ (12-10 for cg models, I think)
            p0 = parms[k][0]
            p1 = parms[k][1]
            #params combination
            x_10 =  math.pow(x, 10)
            #x_10 =  math.pow(x, 6)
            x_12 =  math.pow(x, 12)

            # potential evaluation (formula below has been validated by direct comparison with a traj of 100 frames of a 2-particle system)
            u = - p0/x_10 + p1/x_12
            #u = 0.0000
            U += u

            #print k,x,pot_type[k],p0,p1,u, U

        else:
            print "********************************************"
            print "Code unavailable for this type of potential:",pot_type[k]
            print "********************************************"
            exit()

        if printf:
            strngdst += '{:<30.25f}{:<s}'.format(x, " ")
            strngpot += '{:<30.25f}{:<s}'.format(u, " ")

    if printf:
        with open(outdir+'/dists_ts.txt','a') as distf:
            distf.write('{:<s}\n'.format(strngdst))
        with open(outdir+'/pots_ts.txt','a') as potf:
            potf.write('{:<s}\n'.format(strngpot))

        distf.close()
        potf.close()

    return U

def nb_parms_from_atmtypes(dict_atm_types, mix_rule, n_types):
    """ Returns a 2-dim array of params corresponding to pair-interactions

    Args:
        dic_atm_types (dictionary): Dict. of atom-type and parameters. Example:
            LJ interaction gives entry {'CAK': [3,0.0,1.6777216e-05]}. 3 is the
            position where CAK appears in [ atomtypes ] and so CAK is the type
            3. The other two numbers are c6 and c12 (or c10 and c12) if
            mix_rule=1 is used. In any case I have only two params because only
            LJ are possible here.
        mix_rule (list of int): Type of GROMACS mixing rule for combining
            parameters.
        n_types (int): Number of distinct atom-types.

    """


    parms_mat = [[1000000000 for i in range(n_types)] for j in range(n_types)]
    c_homo = 0
    c_hetero = 0

    for k1, v1 in dict_atm_types.iteritems():
        for k2, v2 in dict_atm_types.iteritems():
            i = v1[0]
            j = v2[0]
            if mix_rule[0] == 1:
                if i == j: #equivalent to say that I have the same type, for example 3 = 3, 'CAK' = 'CAK'
                    #homo-params
                    c_homo += 1
                    p0 = v1[1]
                    p1 = v1[2]
                    parms_mat[i][j] = [p0, p1]
                else:
                    #hetero-params
                    c_hetero += 1
                    p0 = math.sqrt(v1[1]*v2[1])
                    p1 = math.sqrt(v1[2]*v2[2])
                    parms_mat[i][j] = [p0, p1]
            elif mix_rule == 2:
                print "Mixing rule not defined."
                exit()
            elif mix_rule == 3:
                print "Mixing rule not defined."
                exit()

    print "Number of homo and hetero (divide by 2 if isotropic pots) combinations are:",c_homo,c_hetero
    return parms_mat

def all_pairs_in_nl(neighbors):
    #build pairs of atoms in contacts from neighbors list. The neighbors format is, for each "row": first element = central atom
    #all the other elements = neighbors of central atom.
    uniq_pairs_from_all_neigh = []
    un_pairs = []

    for nn in neighbors:
        pairs_from_one_neigh = [[nn[0],int(x)] for x in nn[1:]]

        for p in pairs_from_one_neigh:
            p.sort() # here I sort the two elements of one pair but not pairs among themselves
            un_pairs.append(p)

    uniq_pairs_from_all_neigh = [list(item) for item in set(tuple(row) for row in un_pairs)]

    return uniq_pairs_from_all_neigh

def typize_entry(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_gmxtop(filen):

    try:
        open(filen, "r")
    except:
        print "Error: file %s not found" % filen
        exit()

    top_sections = {
            '[ defaults ]': False,
            '[ atomtypes ]': False,
            '[ moleculetype ]': False,
            '[ atoms ]': False,
            '[ bonds ]': False,
            '[ angles ]': False,
            '[ dihedrals ]': False,
            '[ pairs ]': False,
            '[ exclusions ]': False,
            '[ system ]': False,
            '[ molecules ]': False
            }

    top_headers= {} #the key is the section name and the value is a list

    top_entries = {
            '[ defaults ]': [],
            '[ atomtypes ]': [],
            '[ moleculetype ]': [],
            '[ atoms ]': [],
            '[ bonds ]': [],
            '[ angles ]': [],
            '[ dihedrals ]': [],
            '[ pairs ]': [],
            '[ exclusions ]': [],
            '[ system ]': [],
            '[ molecules ]': []
            }


    CC = ";" #comment character: for start of comments
    n_sec = 0
    title = ""
    with open(filen,"r") as inpfile:
        for line in inpfile:
            lin = line.strip() #assumes strip removes trailing and leading whitespaces
            if len(lin) > 0:
                #check if a section delimiter has been found
                if lin[0] == "[" and lin[-1] == "]":
                    n_sec += 1
                    curr_sec = lin
                    if not top_sections[lin]:
                        top_sections[lin] = True
                else:
                    if lin[0] == CC and n_sec == 0:
                        title += line
                    elif lin[0] == CC and n_sec != 0:
                        top_headers[curr_sec] = lin[1:].split()
                    elif lin[0] != CC and n_sec !=0 :
                        right_types = []
                        for s in lin.split():
                            #The .top format is fixed and therefore there should be a strict control on it.
                            #Use of typize_entry below assumes the .top is without errors/typos.
                            typized = typize_entry(s)
                            right_types.append(typized)
                        top_entries[curr_sec].append(right_types)
                    else:
                        print "Possible errors in .top file:", line
                        exit()


    return top_sections, top_headers, top_entries

def parse_defaults_section(def_sect):
    pot_type = []
    mix_rule = []
    gen_pair = []
    others = []

    for el in def_sect:
        pot_type.append(el[0])
        mix_rule.append(el[1])
        gen_pair.append(el[2])
        others.append(el[3:])

    return pot_type, mix_rule, gen_pair, others

def dictionarize(key, value):
    # the dictionary in the case of the atom types is for example
    # {'CAK:' [i,p0,p1]} where CAK indicates a Calpha-Lysine
    # i its (unique) position in the section [ atomtypes ]
    # p0 and p1 the parameters for the interaction potential
    if len(key) != len(value):
        print "Number of keys and number of values differ:",len(key),len(value)
        exit()

    dictionary = {}
    for k in range(len(key)):
        dictionary[key[k]] = [k]
        dictionary[key[k]].extend(value[k])

    return dictionary

def parse_atmtyp_section(atmtyp_sect):
    attyp = []
    mass = []
    chrg = []
    chn = []
    parms = []

    for el in atmtyp_sect:
        attyp.append(el[0])
        mass.append(el[1])
        chrg.append(el[2])
        chn.append(el[3])
        parms.append(el[4:])

    return attyp, mass, chrg, chn, parms

def parse_atoms_section(atoms_sect):
    nr = []
    typ = []
    resnr = []
    res = []
    atom = []
    cgnr = []
    charge = []
    weight = []

    for el in atoms_sect:
        nr.append(el[0])
        typ.append(el[1])
        resnr.append(el[2])
        res.append(el[3])
        atom.append(el[4])
        cgnr.append(el[5])
        charge.append(el[6])
        weight.append(el[7])

    return nr, typ, resnr, res, atom, cgnr, charge, weight

def parse_pairs_section(pairs_sect):
    at1 = []
    at2 = []
    pot_type = []
    parms = []

    for el in pairs_sect:
        at1.append(el[0])
        at2.append(el[1])
        pot_type.append(el[2])
        parms.append(el[3:])


    return at1, at2, pot_type, parms

def parse_moltype_section(moltype_sect):
    molnam = []
    nrexcl = []

    for el in moltype_sect:
        molnam.append(el[0])
        nrexcl.append(el[1])

    return molnam, nrexcl

def check_atmtyp_sect(attyp, mass, chrg, chn, parms, pot_type):
    # 1.
    tmp = [len(attyp), len(mass), len(chrg), len(chn), len(parms)]
    if not tmp.count(tmp[0]) == len(tmp):
        print ".top file is potentially wrong. Check that attyp, mass, chrg, chn are properly defined in the atomtypes section."
        print len(attyp), len(mass), len(chrg), len(chn), len(parms)
        exit()

    # 2.
    for c, p in enumerate(parms):
        n_prms = dict_pot[pot_type[0]][1]
        if len(p) != n_prms:
            print "Error: pair interaction num %d has %d parameters in .top file, while expected are %d" % (c+1, len(p), n_prms)
            print attyp[c], mass[c], chrg[c], chn[c], parms[c]
            exit()
    return

def check_atoms_sect(nr, typ, resnr, res, atom, cgnr, charge, weight):
    # 1.
    tmp = [len(nr), len(typ), len(resnr), len(res), len(atom), len(cgnr), len(charge), len(weight)]
    if not tmp.count(tmp[0]) == len(tmp):
        print ".top file is potentially wrong. Check that the entries in [ atoms ] section are properly defined."
        print len(nr), len(typ), len(resnr), len(res), len(atom), len(cgnr), len(charge), len(weight)
        exit()

    return

def check_pairs_sect(at1, at2, pot_type, parms):
    # 1.
    tmp = [len(at1), len(at2), len(pot_type), len(parms)]
    if not tmp.count(tmp[0]) == len(tmp):
        print ".top file is potentially wrong. Check that at1, at2 and pot_type are defined in the pairs section."
        exit()

    # 2.
    for c, p in enumerate(parms):
        n_prms = dict_pot[pot_type[c]][1]
        if len(p) != n_prms:
            print "Error: pair interaction num %d has %d parameters in .top file, while expected are %d" % (c+1, len(p), n_prms)
            print at1[c], at2[c], pot_type[c], parms[c]
            exit()
    return

def check_pairs_arr(pairsidx_ps, pairs_sect):
    if len(pairsidx_ps) != len(pairs_sect):
        print "Error: array pairsidx_ps and pairs_sect do not have the same size",len(pairsidx_ps), len(pairs_sect)
        exit()
    else:
        print "Section [ pairs ] has %d entries" % len(pairs_sect)
        print"pairsidx_ps:"
        print(pairsidx_ps)
        print ""
    return

def check_nl_arr(nl_ps, nl_atmtyp, nl_pairs):
    if (len(nl_ps)+len(nl_atmtyp)) != len(nl_pairs):
        print "Error: the sum of %d and %d is not equal to the number of pairs %d in the nl" % (len(nl_ps), len(nl_atmtyp),len(nl_pairs))
        exit()
    return

def  exclude_pairs(nrexcl, nl_atmtyp):
    n_pairs_atmtyp_w_excl = 0
    nb_pairs_excluded = 0
    nl_atmtyp_w_excl = []
    nl_atmtyp_excluded = []
    for p in nl_atmtyp:
        if (math.fabs(p[1]-p[0]) <= nrexcl[0]):
            nb_pairs_excluded += 1
            nl_atmtyp_excluded.append(p)
        else:
            n_pairs_atmtyp_w_excl += 1
            nl_atmtyp_w_excl.append(p)

    return nl_atmtyp_w_excl, nl_atmtyp_excluded

def check_nl_arr2(n_pairs_atmtyp_excluded, n_pairs_atmtyp_w_excl, n_pairs_atmtyp):

    if n_pairs_atmtyp_excluded + n_pairs_atmtyp_w_excl != n_pairs_atmtyp:
        print "nb pairs excluded + included is not equal to n_pairs_atmtyp:",n_pairs_atmtyp_excluded, n_pairs_atmtyp_w_excl, n_pairs_atmtyp
        exit()

    return

def upd_on_max_displacement(nca, traj, frame_at_upd):

    drmax=0.0
    drmax2=0.0
    for k in range(nca):
        r = traj.xyz[frame,:,:]
        dx = r[k][0] - traj.xyz[frame_at_upd[-1]][k][0]
        dy = r[k][1] - traj.xyz[frame_at_upd[-1]][k][1]
        dz = r[k][2] - traj.xyz[frame_at_upd[-1]][k][2]
        dr = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dr > drmax:
            drmax2 = drmax
            drmax = dr
        else:
            if dr > drmax2:
                drmax2 = dr

    # update condition
    update_dyn = False
    if (drmax + drmax2) > drcut:
        update_dyn = True

    return update_dyn

def check_arr_sizes_are_equal(*list_args):
    xxx = [arg for arg in list_args]
    if xxx.count(xxx[0]) != len(xxx):
        print "ERROR: arrays given as arguments have not the same size.", xxx
        exit()

    return

def check_if_dist_longer_cutoff(dist_array, strr, cutoff):
    if any(x>cutoff for x in dist_array):
        print "ERROR: distance in array %s larger than cutoff. It might be due to a too large nstlist." % (strr)
    return

def print_atmtypes_info(dict_atm_types, parms_mt, size):
    print "Dictionary of atom types:"
    print(dict_atm_types)
    print ""
    print "Matrix of nb interactions betwern type:"
    for i in range(size):
        for j in range(size):
            print ('[%15.13f, %15.13f]') % (parms_mt[i][j][0], parms_mt[i][j][1]),
        print ""
    print""

    return

def parse_and_return_relevant_parameters(topf, outdir=None):
    """ Parse a directory's top-file and determine function types

    Read a directory and read the topfiles in order to determine the nonbonded
    interactions; the native interaction and the generic CA interactions.

    Args:
        topf (str): Name of the GROMACS topology file.
        outdir (str): Name of directory to store log-file. Default of None is
        no log file written.

    Attributes:
        top_entries (dictionary): keys are GROMACS specific section tag.
            Elements are typically a list. Example: GROMACS tag `[ atoms ]` has
            the entry `atoms` with a list of atoms.

    returns:
        numeric_atmtyp (list): List of each type
        pairsidx_ps (list): Elements are [at1, at2] corresponding to atom
            indices for native interactions. with exactly the same order as in
            section `[ pairs ]` of file the topf file.
        all_ps_pairs (list):
    """

    if outdir is not None:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        logf = "log.txt"
        sys.stdout = open(outdir+"/"+logf, 'w') #after any statement that reads from stdin

    if topf is not None:
        top_sect, top_head, top_entries = parse_gmxtop(topf)

        def_sect = top_entries['[ defaults ]']
        atmtyp_sect = top_entries['[ atomtypes ]']
        atoms_sect = top_entries['[ atoms ]']
        pairs_sect = top_entries['[ pairs ]']
        moltype_sect = top_entries['[ moleculetype ]']

        # 1. parse [ defaults ] section, generate dictionary of types and nb pot params
        pot_type1_, mix_rule, gen_pair, others = parse_defaults_section(def_sect)

        # 2. parse [ atomtypes ] section
        attyp, mass, chrg, chn, atprops = parse_atmtyp_section(atmtyp_sect)
        n_types = len(attyp)
        check_atmtyp_sect(attyp, mass, chrg, chn, atprops, pot_type1_)
        dict_atm_types = dictionarize(attyp, atprops)
        parms_mt = nb_parms_from_atmtypes(dict_atm_types, mix_rule, n_types)

        print_atmtypes_info(dict_atm_types, parms_mt, n_types)

        # 3. parse [ atoms ] section
        # I need to know which type each atom belongs to, for the nb potentials later.
        # Below: only nr and typ are "fixed" according to the manual.
        nr, typ, resnr, res, atom, cgnr, charge, weight = parse_atoms_section(atoms_sect)
        check_atoms_sect(nr, typ, resnr, res, atom, cgnr, charge, weight)

        # Map the typ (name) to the number that represents the atom type, using dict_atm_types
        numeric_atmtyp = []
        for nam in typ:
            numeric_atmtyp.append(dict_atm_types[nam][0])

        print "numeric_atmtyp array:"
        print numeric_atmtyp
        print ""

        # 4. parse [ pairs ] section
        # pairsidx_ps is a list [at1, at2] with exactly the same order as in section [ pairs ] of file .top
        at1, at2, pot_type2_, parms2_ = parse_pairs_section(pairs_sect)
        check_pairs_sect(at1, at2, pot_type2_, parms2_)
        len_pairs_sec = len(at1)
        pairsidx_ps = [[at1[i], at2[i]] for i in range(len_pairs_sec)]
        all_ps_pairs = [[at1[i]-1, at2[i]-1] for i in range(len_pairs_sec)]
        check_pairs_arr(pairsidx_ps, pairs_sect)

        # 5. parse [ moleculetype ] section
        molnam, nrexcl = parse_moltype_section(moltype_sect)

    else:
        print "No force field parameters specified. Energy calculation impossible."

    return dict_atm_types, numeric_atmtyp, pairsidx_ps, all_ps_pairs, pot_type1_, pot_type2_, parms_mt, parms2_, nrexcl

######### END PARSE TOP FILE ##################################################

def parse_traj_neighbors(traj, numeric_atmtyp, pairsidx_ps, all_ps_pairs, pot_type1_, pot_type2_, parms_mt, parms2_, nrexcl, rcut1=2.0, cons_frq_upd=True, nl_updates=0, nstlist=20):
    """ Parse a traj file and return the

    Generating Neighbor List
    ------------------------
    The neighbors list (nl) is calculated at least once. The nl is (assumed) to
    be calculated in the first frame (init_frame) and then every nstlist frame
    afterwards if cons_frq_upd=True, otherwise with a mist of nstlist and max
    displacement criteria.

    This function makes heavy use of PySPH. PySPH defines a particle as a set
    of atoms. IMPORTANT: get_particle_array has attributes x, y, z already
    built-in, where xyz are 1-D arrays (lists or numpy are okay) that list the
    xyz coordinates of the atoms belonging to the "particle array" or species
    or whatever. Other properties can be defined using prop=value. The argument
    'h' is important argument controlling the linked-cells list. In fact,
    nnps.LinkedListNNPS(dim=3, particles=[pa], radius_scale=rcut2) means that
    radius_scale equals rcut2*h (this multiplication is done internally).

    Syntax to get the nearest neighbors: nps.get_nearest_particles(src_index=0,
    dst_index=0, d_idx=0, nbrs) where src_index and dst_index indicate the
    "particle" indices (ints) of source and destination.

    d_idx is the integer that denotes the atom (in the destination, I believe)
    around which neighbors (in the source) will be searched. The call will
    return, for the d_idx atom of the dst_index "particle", nearest neighbors
    from the src_index "particle".

    frame_at_upd is an important array for understanding how many times the nl
    search occurs. This array stores the frames at which the nl search function
    is called.

    Generating Potential Lists:
    ---------------------------
    Useful info: The function all_pairs_in_nl: 1) eliminates sublist duplicates;
    2) orders within each sublist.
    Example: [[1,10],[7,3]],[1,10]] --> [[1,10],[3,7]]].

    Note: the relative order of the different sublist is
    unchanged. Based on pairsidx_ps, the array of atom pairs indices read from
    the .top file, two important lists are built: nl_ps and nl_atmtyp.

    Remember: The pairsidx_ps numbering of pairs starts from 1. nl_ps and
    nl_atmtyp numbering of pairs start from 0.

    Information on the exclusions is treated soon after building nl_ps and
    nl_atmtyp. Exclusions affect the bonded pairs and the non-bonded pairs
    interacting via [ atomtypes ] parameters, but not the pairs specified in the
    [ pairs ] section.

    nexcl is the parameter that defines the "bonded exclusions". There is also
    the section [ exclusions ] which regulates the non-bonded exclusions. With
    this I mean that all pairs specified in [ exclusions ] are excluded from the
    calculation of the general non-bonded i.e. that from [ atomtypes ] The pairs
    specified in [ pairs ] are not affected by [ exclusions ]!!! params1 array,
    which is the one of the generic type-specific interactions.

    By construction (see for below) parms1 and pot_type1 will be lists of the
    same size of nl_atmtyp and "well ordered" or "corresponding". It is
    important that parms1 and pot_type1 are "well ordered" also with the
    distance array dist_nl_atmtyp. This is true because I use
    md.compute_distances(..., dist_nl_atmtyp).

    Similar argument for parms2: the ordering is controlled below and consistent
    with md.compute_distances(struct, nl_ps)

    Args:
        traj (mdtraj.Trajectory): Trajectory to parse
        numeric_atmtyp (list): See parse_and_return_relevant_parameters().
        pairsidx_ps (list): See parse_and_return_relevant_parameters().
        all_ps_pairs (list): See parse_and_return_relevant_parameters().

    """

    rcut2 = rcut1 + 0.25 #gromacs manual page 93, v 4.5.4
    drcut = (rcut2 - rcut1)

    frame_at_upd = []
    init_frame = 0  #min 0 max nfr-1

    print "#################################################################"
    print "rcut used in nl search is:",rcut2," nm"
    if not cons_frq_upd:
        print "nl updates starting at frame %d every nstlist, if large displacements do not occur: %d" % (init_frame, nstlist)
    else:
        print "nl updates every nstlist %d is enforced:" % (nstlist)
    print "#################################################################"
    print ""

    update_dyn = False
    update_nstlist = False
    update = False

    nfr = traj.n_frames
    all_nl_ps = []
    all_nl_atmtyp_w_excl = []

    all_parms1 = []
    all_pot_type1 = []

    all_parms2 = []
    all_pot_type2 = []

    for frame in range(init_frame,nfr):
        #first index is 0 because struct = traj[frame]
        struct = traj[frame]
        rx = struct.xyz[0,:,0]
        ry = struct.xyz[0,:,1]
        rz = struct.xyz[0,:,2]

        check_arr_sizes_are_equal(len(rx), len(ry), len(rz))
        nca = len(rx)

        if not cons_frq_upd:
            if frame >= 1:
                update_dyn = upd_on_max_displacement(nca, traj, frame_at_upd)

        snap = frame - init_frame
        if snap%nstlist == 0:
            update_nstlist = True
        else:
            update_nstlist = False

        # update_dyn has the priority over update_nstlist
        update = False
        if update_dyn:
            update = True
        else:
            update = update_nstlist

        if update:

            print "Neighbor search",nl_updates

            pa = get_particle_array(name='prot', x=rx, y=ry, z=rz, h=1.0) #h=1.0 must not be changed as it affects radius_scale in nnps.LinkedListNNPS

            nps = nnps.LinkedListNNPS(dim=3, particles=[pa], radius_scale=rcut2)
            #nps = nnps.SpatialHashNNPS(dim=3, particles=[pa], radius_scale=rcut2)

            src_index = 0
            dst_index = 0
            neighbors = []
            tot = 0

            for i in range(nca):
                nbrs = UIntArray()
                nps.get_nearest_particles(src_index, dst_index, i, nbrs)
                tmp = nbrs.get_npy_array().tolist()

                # patch
                if tmp.count(i) > 1:
                    print "Problems: nbrs has multiple occurrences of residue %d, which might indicate something wrong in pysph (for certain radius_scale values)" %(i)
                    print "Patching this problem by assuming that residue %d has no neighbors other than itself" % (i)
                    tmp = [i]
                else:
                    pass

                nneigh = len(tmp)
                tot += nneigh

                #I do not want neighbors that are only the i-th particle (i.e. self)
                #I want the "central" atom as the first element: THIS IS FUNDAMENTAL FOR HOW all_pairs_in_nl WORKS
                if (nneigh > 1):
                    tmp.remove(i)
                    tmp = [i] + tmp
                    neighbors.append(tmp[:])

            len_nl_arr = len(neighbors)
            nl_updates += 1
            frame_at_upd.append(frame)

    ####### END NEIGHBOR LIST CALCULATION ####################################

            ### VERY IMPORTANT NOTE: if len_nl_arr >= 1 is within the same block that updates the nl
            ### and this is clear: all lists that derive from neighbors (which is the neighbor list)
            ### must be calculated only if neighbors changes. Otherwise, it is a waste of time.
            if len_nl_arr >= 1:

                # I improved the speed of all_pairs_in_nl a bit. Maybe something smarter can be still done to improve performance.
                nl_pairs = all_pairs_in_nl(neighbors)

                # nl_ps and nl_atmtyp are numbered from 0. pairsidx_ps is numbered from 1
                # A note here: removing from nl_pairs ALL pairwise interactions declared in the .top file is fine even though,
                # at a given frame, not all declared pairwise could be present. However, because the "absent" ones (those not
                # found at nl search time) are not interacting with potentials coming from the [ atomtypes ] section, it means
                # that the operation of removing all_ps_pairs from nl_pairs (see below) is fine.
                nl_pairs_less_all_ps = [list(item) for item in (set(tuple(row1) for row1 in nl_pairs) - set(tuple(row2) for row2 in all_ps_pairs))]
                nl_atmtyp = nl_pairs_less_all_ps
                nl_ps = [list(item) for item in (set(tuple(row1) for row1 in nl_pairs) - set(tuple(row2) for row2 in nl_pairs_less_all_ps))]


                ##### I KEEP IT FOR NOW TO DEBUG but it is not necessary and does not ######
                ##### hurt the code: in fact, later I order the pot parms based on    ######
                ##### the pair lists and distances                                    ######
                #nl_ps = sorted(nl_ps, key = lambda x: (x[0], x[1]))
                #nl_atmtyp = sorted(nl_atmtyp, key = lambda x: (x[0], x[1]))
                #############################################################

                check_nl_arr(nl_ps, nl_atmtyp, nl_pairs)
                n_pairs_ps = len(nl_ps)
                n_pairs_atmtyp = len(nl_atmtyp)
                n_pairs = len(nl_pairs)

                # Exclusions, check for pairs to be excluded
                nl_atmtyp_w_excl, nl_atmtyp_excluded = exclude_pairs(nrexcl, nl_atmtyp)
                n_pairs_atmtyp_w_excl = len(nl_atmtyp_w_excl)
                n_pairs_atmtyp_excluded = len(nl_atmtyp_excluded)
                check_nl_arr2(n_pairs_atmtyp_excluded, n_pairs_atmtyp_w_excl, n_pairs_atmtyp)

                # This part is very important: It is meant to build "well ordered" parms1,
                # the parameters that pertain to each pair interacting with generic potentials
                # More about this in at the beginning of the section.
                pot_type1 = []
                parms1 = []
                for p in nl_atmtyp_w_excl:
                    i = numeric_atmtyp[p[0]]
                    j = numeric_atmtyp[p[1]]
                    parms1.append(parms_mt[i][j])
                    pot_type1.append(pot_type1_[0])

                #Construct the "well ordered" parms2 array from the one parsed from .top file.
                # More about this in at the beginning of the section.
                pot_type2 = []
                parms2 = []
                for pp in nl_ps:
                    p = [pp[0] + 1, pp[1] + 1]
                    idx = pairsidx_ps.index(p)  #pairsidx_ps is what I have in the .top file, i.e. the maximum number of pairs that nl_ps can be
                    parms2.append(parms2_[idx])
                    pot_type2.append(pot_type2_[idx])

            else:
                print "No pairs. Array neighbors has only %d elements" % (len_nl_arr)
                exit()
    ######## END ORGANIZE NL ARRAYS AND ORDER POT PARAMS BASED ON NL ARRAYS ############

        all_nl_ps.append(nl_ps)
        all_nl_atmtyp_w_excl.append(nl_atmtyp_w_excl)

        all_parms1.append(parms1)
        all_pot_type1.append(pot_type1)

        all_parms2.append(parms2)
        all_pot_type2.append(pot_type2)

    return all_nl_ps, all_nl_atmtyp_w_excl, all_parms1, all_pot_type1, all_parms2, all_pot_type2, rcut2

def compute_energy_slow(traj, all_nl_ps, all_nl_atmtyp_w_excl, all_parms1, all_pot_type1, all_parms2, all_pot_type2, rcut2):
    """ Compute non-bonded energies for a trajectory given parameters

    Using the pre-computed table values of a structure, the non-bonded energy
    for a trajectory is computed. This includes both the gaussian native
    interactions as well as the LJ-12-10 type non-native interactions.

    There has to be a STRICT CORRESPONDENCE between each element of
    dist_nl_atmtyp, pot_type1, parms1 or dist_nl_ps, pot_type2, parms2.

    This correspondence has been achieved by ordering parms and pot_type arrays
    as the distance arrays, that is as the neighbor list arrays from which the
    distances are calcualated (using mdtraj). The below code is executed at
    every frame, given the nl lists, params etc... that do not change between nl
    searches (above). The variable struct is the mdtraj snapshot relative to the
    current frame and is used in md.compute_distances.
    """

    ######## CALCULATE DISTANCES AND ENERGIES FOR EACH SNAPSHOT, GIVEN THE NL LISTS BEFORE ############

    all_Enb = []

    nfr = traj.n_frames
    for frame in range(nfr):
        struct = traj[frame]
        nl_ps = all_nl_ps[frame]
        nl_atmtyp_w_excl = all_nl_atmtyp_w_excl[frame]

        parms1 = all_parms1[frame]
        pot_type1 = all_pot_type1[frame]

        parms2 = all_parms2[frame]
        pot_type2 = all_pot_type2[frame]

        n_pairs_ps = len(nl_ps)
        n_pairs_atmtyp_w_excl = len(nl_atmtyp_w_excl)

        Enb = 0.0
        Enb_pair_sec = 0.0
        Enb_atmtyp_sec = 0.0

        if n_pairs_ps > 0:
            dist_nl_ps = md.compute_distances(struct, nl_ps)
            dist_nl_ps = dist_nl_ps.ravel()
            check_if_dist_longer_cutoff(dist_nl_ps, 'dist_nl_ps', rcut2)

            check_arr_sizes_are_equal(len(dist_nl_ps),len(pot_type2),len(parms2))
            Enb_pair_sec = calc_nb_ene(dist_nl_ps, pot_type2, parms2, False)

        if n_pairs_atmtyp_w_excl > 0:
            dist_nl_atmtyp = md.compute_distances(struct, nl_atmtyp_w_excl)
            dist_nl_atmtyp = dist_nl_atmtyp.ravel()
            check_if_dist_longer_cutoff(dist_nl_atmtyp, 'dist_nl_atmtyp (w exclusions)', rcut2)

            check_arr_sizes_are_equal(len(dist_nl_atmtyp), len(pot_type1),len(parms1))
            Enb_atmtyp_sec = calc_nb_ene(dist_nl_atmtyp, pot_type1, parms1, False)

        Enb = Enb_pair_sec + Enb_atmtyp_sec

        print "Non bonded energy of frame %d is %30.25f (pairs and atmtyp contributions are %30.25f %30.25f):" % (frame, Enb, Enb_pair_sec, Enb_atmtyp_sec)
        all_Enb.append(Enb)

    return all_Enb

def prep_compute_energy_fast(traj, all_nl_ps, all_nl_atmtyp_w_excl, numeric_atmtyp, pairsidx_ps, params_mt_noeps, parms2_, pot_type1_, pot_type2_, rcut2):
    """ Prepare the necessary arrays to compute the potential energy quickly

    """
    all_nonbonded_eps_idxs = []
    all_nonbonded_factors = []
    all_pairwise_eps_idx = []
    all_pairwise_factors = []

    nfr = traj.n_frames
    for frame in range(nfr):
        struct = traj[frame]
        nl_ps = all_nl_ps[frame]
        nl_atmtyp_w_excl = all_nl_atmtyp_w_excl[frame]

        pot_type1 = []
        parms1 = []
        Enb_pair_idxs = []
        Enb_atmtyp_idxs = []
        for p in nl_atmtyp_w_excl:
            i = numeric_atmtyp[p[0]]
            j = numeric_atmtyp[p[1]]
            parms1.append(params_mt_noeps[i][j])
            pot_type1.append(pot_type1_[0])
            Enb_atmtyp_idxs.append([i,j])

        #Construct the "well ordered" parms2 array from the one parsed from .top file.
        # More about this in at the beginning of the section.
        pot_type2 = []
        parms2 = []
        for pp in nl_ps:
            p = [pp[0] + 1, pp[1] + 1]
            idx = pairsidx_ps.index(p)  #pairsidx_ps is what I have in the .top file, i.e. the maximum number of pairs that nl_ps can be
            parms2.append(parms2_[idx])
            pot_type2.append(pot_type2_[idx])
            Enb_pair_idxs.append(idx)

        n_pairs_ps = len(nl_ps)
        n_pairs_atmtyp_w_excl = len(nl_atmtyp_w_excl)


        if n_pairs_ps > 0:
            dist_nl_ps = md.compute_distances(struct, nl_ps)
            dist_nl_ps = dist_nl_ps.ravel()
            check_if_dist_longer_cutoff(dist_nl_ps, 'dist_nl_ps', rcut2)

            check_arr_sizes_are_equal(len(dist_nl_ps),len(pot_type2),len(parms2))
            Enb_pair_sec = calc_nb_ene_fast(dist_nl_ps, pot_type2, parms2)
        else:
            # append a single zeros
            Enb_pair_sec = np.zeros(1)
            Enb_pair_idxs = [[0,0]]


        assert np.shape(Enb_pair_sec)[0] == len(Enb_pair_idxs)

        if n_pairs_atmtyp_w_excl > 0:
            dist_nl_atmtyp = md.compute_distances(struct, nl_atmtyp_w_excl)
            dist_nl_atmtyp = dist_nl_atmtyp.ravel()
            check_if_dist_longer_cutoff(dist_nl_atmtyp, 'dist_nl_atmtyp (w exclusions)', rcut2)

            check_arr_sizes_are_equal(len(dist_nl_atmtyp), len(pot_type1),len(parms1))
            Enb_atmtyp_sec = calc_nb_ene_fast(dist_nl_atmtyp, pot_type1, parms1)
        else:
            Enb_atmtyp_sec = np.zeros(1)
            Enb_atmtyp_idxs = [[0,0]]

        # check number of pairs match number of parameters
        assert len(Enb_atmtyp_idxs) == len(Enb_atmtyp_sec)
        assert len(Enb_pair_idxs) == len(Enb_pair_sec
        )
        all_nonbonded_factors.append(Enb_atmtyp_sec)
        all_nonbonded_eps_idxs.append(Enb_atmtyp_idxs)
        all_pairwise_factors.append(Enb_pair_sec)
        all_pairwise_eps_idx.append(Enb_pair_idxs)
    # cneck number of frames all match
    try:
        assert len(all_nonbonded_factors) == nfr
        assert len(all_nonbonded_eps_idxs) == nfr
        assert len(all_pairwise_factors) == nfr
        assert len(all_pairwise_eps_idx) == nfr
    except:
        print len(all_nonbonded_factors)
        print len(all_nonbonded_eps_idxs)
        print len(all_pairwise_factors)
        print len(all_pairwise_eps_idx)
        raise
    return all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors

def compute_energy_fast(nonbonded_eps_matrix, pairwise_eps_list, all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors):
    """ A much faster version of the energy calculation

    Instead of using parmeter and pair index lists in order to compute the potential energy as in compute_energy_slow(), use just the new epsilons, and precompute the potential energies divided by the epsilons. Then it becomes a simple matter of multitplying the pre-computed no-epsilon potentials with the epsilons.

    This has to be done in list form as the number of potential calculations for each frame varies. This avoids the necessity of loading every single distance coordinate and improves the memory scaling to N instead of N^2. This does however increase computation time required as we can no longer use built in numpy methods, but the tradeoff is worth it.

    Assuming you have an array of M epsilons with N frames.

    args:
        epsilons (array of float): M length array  of the parameters used in the
            optimization procedure.
        all_frame_params (list of int): Length N where each entry is an index or
            set of indices corresponding to the parameters to use.
        all_frame_precomputed (list of float): Length N where each entry is the precomputed potential energy divided by the epsilon values.

    returns:
        U (array of floats): Length M where each entry is the potential energy
            for each frame.

    """

    print pairwise_eps_list

    n_frames = len(all_nonbonded_eps_idxs)
    U = np.zeros(n_frames)
    for i_frame in range(n_frames):
        nonbonded_idxs = all_nonbonded_eps_idxs[i_frame]
        nonbonded_factors = all_nonbonded_factors[i_frame]
        pairwise_idxs = all_pairwise_eps_idx[i_frame]
        pairwise_factors = all_pairwise_factors[i_frame]

        n_nonbonded = np.shape(nonbonded_idxs)[0]
        n_pairwise = np.shape(pairwise_idxs)[0]

        nonbonded_eps = np.zeros(n_nonbonded)
        pairwise_eps = np.zeros(n_pairwise)
        for i_nb in range(n_nonbonded):
            nonbonded_eps[i_nb] = nonbonded_eps_matrix[nonbonded_idxs[i_nb][0]][nonbonded_idxs[i_nb][1]]
        for i_pw in range(n_pairwise):
            pairwise_eps[i_pw] = pairwise_eps_list[pairwise_idxs[i_pw]]

        nonbonded_energy = np.sum(nonbonded_eps * nonbonded_factors)
        pairwise_energy = np.sum(pairwise_eps * pairwise_factors)
        this_u = nonbonded_energy + pairwise_energy

        print "Native E: %f   Nonbonded E: %f" % (pairwise_energy, nonbonded_energy)
        U[i_frame] = this_u

    return U

def compute_derivative_fast(nonbonded_matrix_depsilons, all_nonbonded_eps_idxs, all_nonbonded_factors):
    """ Computes the gradient of the potential energy

    For a M epsilons with N frames, expect to return a length M list where each entry is an N length array for the m'th component of the derivative at frame N.

    """
    n_frames = len(all_nonbonded_eps_idxs)
    n_nonbonded_eps = np.shape(nonbonded_matrix_depsilons)[0]
    all_derivatives = [[0 for i in range(n_nonbonded_eps)]]
    for i_frame in range(n_frames):
        nonbonded_idxs = all_nonbonded_eps_idxs[i_frame]
        nonbonded_factors = all_nonbonded_factors[i_frame]

        n_nonbonded = np.shape(nonbonded_idxs)[0]
        n_pairwise = np.shape(pairwise_idxs)[0]

        for i_nb in range(n_nonbonded):
            both_indices = nonbonded_idxs[i_nb]
            idx_nb = both_indices[0]
            jdx_nb = both_indices[1]
            all_derivatives[idx][i_frame] = nonbonded_matrix_depsilons[idx, jdx] * nonbonded_factors[i_nb]
            all_derivatives[jdx][i_frame] = nonbonded_matrix_depsilons[jdx, idx] * nonbonded_factors[i_nb]

    return all_derivatives

if __name__ == "__main__":

    topf = "cg_model/topol.top"
    traj = md.load("cg_model/traj.xtc", top="cg_model/conf.gro")[::10]

    all_nl_ps, all_nl_atmtyp_w_excl, parms1, pot_type1, parms2, pot_type2, rcut2 = parse_traj_neighbors(traj, numeric_atmtyp, pairsidx_ps, all_ps_pairs, pot_type1_, pot_type2_, parms_mt, parms2_, nrexcl)

    all_Enb = compute_energy_slow(traj, all_nl_ps, all_nl_atmtyp_w_excl, parms1, pot_type1, parms2, pot_type2, rcut2)
