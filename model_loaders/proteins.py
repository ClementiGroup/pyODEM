""" Loading data for Molecular Dynamics Simulations

Requires package: https://github.com/ajkluber/model_builder

"""
import numpy as np
import mdtraj as md

from pyfexd.model_loaders import ModelLoader
try:
    import model_builder as mdb
except:
    pass

class ProtoProtein(ModelLoader):
    """ subclass of ModelLoaders, super class of all Protein models
    
    The __init__ for most protein models have shared methods. This way, 
    they can all call the same methods with modificaitons of their own 
    later. 
    
    """
    
    def __init__(self, ini_file_name):
        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k
        
        ##remove .ini suffix
        self.model, self.fittingopts = mdb.inputs.load_model(ini_file_name)
        
        if "fret_pairs" in self.fittingopts and not self.fittingopts["fret_pairs"] is None:
            self.fret_pairs = self.fittingopts["fret_pairs"]
        else:
            self.fret_pairs = [None]
        
        self.beta = 1.0 #set temperature
    
    def load_data(self,fname):
        """ Load a data file and format for later use
        
        For Proteins, it uses the self.pairs to load the pair-pair 
        distances for every frame. This is the data format that would be 
        used for computing the energy later.
        
        Args:
            fname(string): Name of a file to load.
        
        Return:
            Array(floats): First index is frame, second index is every 
                pair in the order of pairs.
            
        """
        
        traj = md.load(fname, top=self.model.mapping.topology)
        data = md.compute_distances(traj, self.use_pairs, periodic=False)

        return data    
    
class Protein(ProtoProtein):
    """ Subclass for making a ModelLoader for a Protein Model
    
    Methods:
        See ModelLoader in pyfexd/super_model/ModelLoader
    
    """
    
    def __init__(self, ini_file_name):
        """ Initialize the Langevin model, override superclass
        
        Args:
            ini_file_name: Name of a .ini file to load containing the 
                model information.
        
        Attributes:
            See superclass for generic attributes.
            epsilons(array): Chosen from a specific list of tunable 
                parameters from the .ini file.
        
        """
        
        ProtoProtein.__init__(self, ini_file_name)
            
        # get indices corresponding to epsilons to use
        # Assumes only parameters to change are pair interactions
        self.use_params = np.arange(len(self.model.Hamiltonian._pairs)) #assumes you use all pairs
        self.pairs = self.model.mapping._contact_pairs
        self.use_pairs = []
        for i in self.use_params: #only load relevant parameters
            self.use_pairs.append([self.pairs[i][0].index, self.pairs[i][1].index])
        
        self.epsilons = self.model.fitted_epsilons
    
    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)  
        
        See superclass for full description of purpose.
        Override superclass. Potential Energy is easily calculated since
        for this model, all epsilons are linearly related to the 
        potential energy.
        
        """
        
        #check to see if data is the expected shape for this analysis:
        if not np.shape(data)[1] == np.shape(self.use_params)[0]:
            err_str = "dimensions of data incompatible with number of parameters\n"
            err_str += "Second index must equal number of parameters \n"
            err_str += "data is: %s, number of parameters is: %d" %(str(np.shape(data)), len(self.use_params))
            raise IOError(err_str)
        
        #list of constant pre factors to each model epsilons
        constants_list = [] 
        constants_list_derivatives = []
        for i in self.use_params:
            constants_list.append(self.model.Hamiltonian._pairs[i].dVdeps(data[:,i]) )
            constants_list_derivatives.append(self.model.Hamiltonian._pairs[i].dVdeps(data[:,i])* -1. * self.beta  )
        #compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])
            for i in range(np.shape(epsilons)[0]):
                value = epsilons[i]*constants_list[i]
                total += value * -1. * self.beta

            return total     
        
        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame

            return constants_list_derivatives
        
        return hepsilon, dhepsilon
        
        
class ProteinNonLinear(Protein):
    """ Same as protein, except handles nonlinear H(epsilons)
    
    Just like the class Protein, except it handles a non-linear 
    Hamiltonian function of epsilons. Thusly, only the 
    get_potentials_epsilon is overridden.
    
    """
        
    def get_potentials_epsilon(self, data):
        num_frames = np.shape(data)[0]

        functions_list = []
        dfunctions_list = []
        for i in self.use_params:
            functions_list.append(self.model.Hamiltonian._pairs[i].get_V_epsilons(data[:,i]))
            dfunctions_list.append(self.model.Hamiltonian._pairs[i].get_dV_depsilons(data[:,i]))
        #compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])
            for i in range(np.shape(epsilons)[0]):
                total += functions_list[i](epsilons[i]) * -1. * self.beta
            #total *= -1. * self.beta

            return total     
        
        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame
            scaled_beta = -1. * self.beta
            derivatives_list = [func(epsilons[idx])*scaled_beta for idx,func in enumerate(dfunctions_list)]

            return derivatives_list
        
        return hepsilon, dhepsilon

'''
        def hepsilon(epsilons):
            total_energy = np.zeros(np.shape(data)[0])
            for idx, param in enumerate(self.use_params):    
                self.model.Hamiltonian._pairs[param].set_epsilon(epsilons[idx])
                energy = self.model.Hamiltonian._pairs[param].V(data[:,idx])
                total_energy += energy
            
            return total_energy * self.beta * -1.
        
        
        def dhepsilon(epsilons):
            total_energy = np.zeros(np.shape(data))
            for idx, param in enumerate(self.use_params):    
                self.model.Hamiltonian._pairs[param].set_epsilon(epsilons[idx])
            
                energy = self.model.Hamiltonian._pairs[param].dVdeps(data[:,idx])
                total_energy[:,idx] += energy
            
            total_energy *= self.beta * -1.
            
            gradient = [total_energy[:,idx] for idx in range(np.shape(self.use_params)[0])]
            
            
            #assert len(gradient) == num_frames
            #for term in gradient:
            #    assert np.shape(term)[0] == np.shape(self.use_params)[0]
            return gradient 
        
        
        return hepsilon, dhepsilon
        
'''        

class ProteinAwsem(ProtoProtein):
    def __init__(self, ini_file_name):
        ProtoProtein.__init__(self, ini_file_name) #not implemented Need to do so in future, but will require modelbuilder implementation as well
        self.GAS_CONSTANT_KJ_MOL /= 4.184 #convert to kCal/mol*K
        self.use_frag = False
        self.use_gammas = False
    def add_fragment_memory_params(self, param_path, mem_file, max_frag_length=9, cycle = True, fragment_memory_scale=0.1):
        """ Add fragment memory interactions for fitting """
        
        self.model.add_fragment_memory(self, param_path, mem_file, max_frag_length=max_frag_length, cycle = cycle, fragment_memory_scale=fragment_memory_scale)
        
        self.use_frag = True
        self.epsilons = []
        for potential in self.model.Hamiltonian.fragment_potentials:
            self.epsilons.append(potential.weight)
        self.epsilons = np.array(self.epsilons)
        self.frag_scale = self.model.Hamiltonian.fragment_memory_scale
        
        
    def add_contact_params(self):
        """ Add direct contact interactions for fitting
        
        Only uses the gammas that are present in the model. It has to 
        assign each interaction to a corresponding gamma, as well as 
        collect epsilon values(gammas) that exist. Code is weird, as its 
        more memory intensive than it needs to be so that sorting things 
        can be completed in 2N+210 time. Where N is the number of direct 
        potentials in the model, and 210 is the number of possible 
        unique gammas it has to sort through. N~R**2, where R is 
        number of residues so minimizing the number of times it goes 
        through each list. 
        
        Attributes:
            use_pairs: list of pair interactions, list contains atom 
                indices for interactions.
            gamma: 20x20 gamma-matrix. gm(i,j) = gm(j,i). 
            gamma_indices: Nx2 array of gamma-matrix indices for each 
                parameter
            use_indices: List with X attributes, where X is number of  
                unique gammas. Values are gamma indices used.
            use_params: List with X attributes. Values are gamma values 
                from the gamma-matrix corresponding to indices in 
                use_indices.
            param_assignment: List of len=N, values are the index for 
                each interaction's corresponding gamma in the use_params 
                list.
            param_assigned_indices: List of len=X, 
                param_assigned_indices[i] = [indices], where indices are 
                the index of the N parameters with the 
                gamma indices = use_indices[i]
        
        """
        self.use_gammas = True
        # get indices corresponding to epsilons to use
        # Assumes only parameters to change are pair interactions
        self.use_pairs = []
        array_pairs = self.model.Hamiltonian._contact_pairs
        for i in range(np.shape(array_pairs)[0]):
             self.use_pairs.append(array_pairs[i,:])

        self.gamma_indices = self.model.Hamiltonian._contact_gamma_idxs
        self.gamma = self.model.Hamiltonian.gamma_direct        
        
        #go through gamma, append the parameters to use_params
        self.use_indices = []
        self.use_params = []
        check_index_array = np.zeros((20,20))
        check_index_assignment = []
        #go through list, mark all positions where a potential is found
        for i in range(np.shape(self.gamma_indices)[0]):
            idx = self.gamma_indices[i,0]
            jdx = self.gamma_indices[i,1]
            check_index_array[idx,jdx] = 1 #set to 1 if found
            check_index_assignment.append((idx*20) + jdx) #convert 2-d to 1-d value
            
        #save index if check(i,j) = 1 or check(j, i) = 1
        count = 0
        check_index_conversion_array = np.zeros(400).astype(int)
        check_index_conversion_array -= 1
        for i in range(20): #this potential is always 20x20
            for j in np.arange(i,20):
                if check_index_array[i,j] == 1 or check_index_array[j,i] == 1:
                    self.use_indices.append([i,j])
                    self.use_params.append(self.gamma[i,j])
                    coord1 = (i*20) + j
                    coord2 = (j*20) + i
                    check_index_conversion_array[coord1] = count
                    check_index_conversion_array[coord2] = count
                    count += 1
                    
        #Now we have which sets of gamma are indeed used.
        #Next, assign each interaction to each parameter, for later
        self.param_assignment = [] #each potential has value of parameter index
        self.param_assigned_indices = [[] for i in range(len(self.use_indices))]
        for i in range(len(check_index_assignment)):
            param_idx = check_index_conversion_array[check_index_assignment[i]]
            self.param_assignment.append(param_idx)
            self.param_assigned_indices[param_idx].append(i)
        
        self.epsilons = self.use_params #self.epsilons expected in places
        
        ##### assertion checks #####
        # Check consistency of param_assignment and param_assigned_indices 
        for idx, lst in enumerate(self.param_assigned_indices):
            for index in lst:
                assert self.param_assignment[index] == idx
        
        #Check consistent number of parameters
        assert len(self.param_assigned_indices) == len(self.use_params)
        
    def load_data(self,fname):
        """ Load a data file and format for later use
        
        For Proteins, it uses the self.pairs to load the pair-pair 
        distances for every frame. This is the data format that would be 
        used for computing the energy later.
        
        Args:
            fname(string): Name of a file to load.
        
        Return:
            traj(mdtraj.trajectory): Trajectory object in 3-bead 
                representation. Will be converted to all-atom later.
            
        """
        
        #remarkably, traj object is indexable like an array
        #it is however only a 1-D array
        traj = md.load(fname, top=self.model.mapping.topology)
        

        return traj  
    
    def save_epsilons(self, new_epsilons):
        assert np.shape(new_epsilons)[0] == len(self.use_indices)
        gamma_matrix = self.model.Hamiltonian.gamma_direct
        for idx,eps in zip(self.use_indices, new_epsilons):
            gamma_matrix[idx[0], idx[1]] = eps
            gamma_matrix[idx[1], idx[0]] = eps
        
        for i in range(20):
            for j in range(i+1, 20):
                assert gamma_matrix[i,j] == gamma_matrix[j,i]
    
    def save_fragment_memory_parameters(self, new_eps, name, directory, write=False):
        assert np.shape(new_eps)[0] == np.shape(self.epsilons)[0]
        
        for i in range(np.shape(new_eps)[0]):
            self.model.Hamiltonian.fragment_potentials[i].weight = new_eps[i]
        
        if write:
            self.model.write_new_fragment_memory(directory, name)
            
    def save_debug_files(self, old_eps, new_eps):
        f = open("debug_used_parameters.txt", "w")
        for index in self.use_indices:
            f.write(self.model.Hamiltonian.gamma_residues[index[0]])
            f.write(" - ")
            f.write(self.model.Hamiltonian.gamma_residues[index[1]])
            f.write("\n")
        f.close()
        np.savetxt("debug_old_eps.dat", old_eps)
        np.savetxt("debug_new_eps.dat", new_eps)
        
        diff = new_eps-old_eps
        np.savetxt("debug_diff_eps.dat", diff)
        sort_idxs = np.argsort(np.abs(diff))
        wrt_str = ""
        for idx in sort_idxs:
            res1 = self.model.Hamiltonian.gamma_residues[self.use_indices[idx][0]]
            res2 = self.model.Hamiltonian.gamma_residues[self.use_indices[idx][1]]
            this_str =  "%s - %s   %f\n" %(res1, res2, diff[idx])
            wrt_str = this_str + wrt_str
        
        f = open("debug_sorted_parameters.txt", "w")
        f.write(wrt_str)
        f.close()
    
    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)  
        
        See superclass for full description of purpose.
        Override superclass. Potential Energy is easily calculated since
        for this model, all epsilons are linearly related to the 
        potential energy.
        
        """
        
        constants_list = [] 
        constants_list_derivatives = []
        
        if self.use_gammas == self.use_frag:
            raise IOError("Both gammas and fragment memory are set to: %s" % (str(self.use_gammas)))
            
        #Depending on which flag is on, it will compose constants_list and constants_list_derivatives     
        if self.use_gammas:
            #for potentials, 1st index is frame, 2nd index is potential function
            potentials = self.model.Hamiltonian.calculate_direct_energy(data, total=False)
            assert np.shape(potentials)[1] == len(self.param_assignment)
        
            for indices,param in zip(self.param_assigned_indices,self.use_params):
                constant_value = np.sum(potentials[:,indices], axis=1) / param
                constants_list.append(constant_value)
                constants_list_derivatives.append(constant_value * -1. * self.beta)
        elif self.use_frag:
            potentials = self.model.Hamiltonian.calculate_fragment_memory_potential(data, total=False)
            for idx in range(np.shape(potentials)[0]):
                rescale_constants = potentials[idx,:] / self.epsilons[idx]
                constants_list.append(rescale_constants)
                constants_list_derivatives.append(rescale_constants * -1. * self.beta)
            assert np.shape(self.epsilons)[0] == np.shape(potentials)[0]
            
        #compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(data.n_frames)
            for i in range(np.shape(epsilons)[0]):
                value = epsilons[i]*constants_list[i]
                total += value * -1. * self.beta

            return total     
        
        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame

            return constants_list_derivatives
        
        return hepsilon, dhepsilon  

