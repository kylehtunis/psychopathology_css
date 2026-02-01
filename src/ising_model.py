import numpy as np
import networkx as nx


class IsingSimulation:

    def __init__(self, network, initial_state=None, H=1.0, J=1.0, T=2.26, length=10_000, warmup=1000):
        """
        Initialises the Ising model simulation

        Parameters:

        network: the network of nodes, where each node is a symptom
        initial_state: the initial state of the system (patient's mind). If nothing is provided, the state will be random

        H: the external field applied on the system - stress
        J: the coupling - how much neighbouring emotions want to align
        T: the temperature/ noise - how strong are random swings from one state to the other
        
        length: the amount of time steps to simulate
        warmup: The amount of steps to take before starting the simulation, to allow the system to settle down
        """

        # The network of nodes, that is the brain of the patient
        self.network = network

        # Transform the network into a numpy array for easier computations down the line
        self.adj_matrix = nx.to_numpy_array(self.network)
        
        # The size of the system N
        self.num_nodes = self.adj_matrix.shape[0]
        
        # The physics that govern the system
        self.H = H # The external stressor
        self.T = T # The noise in the system

        # compute the coupling between emotions so that the default behaviour is a patient with a rigid brain
        # whose emotions are strongly coupled
        # This allows to display a sharp phase transition from healthy to depressed if the external stress is high
        if J is None:

            # degrees of all nodes
            degrees = np.sum(self.adj_matrix, axis=1)
            # largest degree
            max_degree = np.max(degrees)
            
            # we want the nodes with largest degrees to be stable against the noise T
            factor = 2
            self.J = (self.T  / max_degree) * factor
            
            # print(f"computed J: {self.J:.4f}")
        
        else:
        
            self.J = J

        # The initial state of the system
        # If an initial state is not provided we should assign randomly a value in {-1, 1} to each node
        # However if we provide an initial state, the warmup is prohibited since we would be deviating from that initial state
        if initial_state is None:

            self.warmup = warmup
            self.current_state = np.random.choice([-1, 1], size=self.num_nodes)
        
        else:
        
            self.warmup = 0
            self.current_state = np.array(initial_state)

        # The time settings
        self.length = length # the number of time steps to perform

        # The records of the evolution of the simulation
        # Storing the state of each node for all time steps
        self.history = np.zeros((self.length, self.num_nodes), dtype=np.int8)

        # Store the state of each node during the warmup period (for covariance calculations)
        self.warmup_states = np.zeros((self.warmup, self.num_nodes), dtype=np.int8)

        # storing the average mood - order parameter, m, at each time step
        self.order_history = np.zeros(self.length)

        # Storing the external field/Stress applied on the system at each point in time
        self.h_history = np.zeros(self.length)

        # Storing the covariance matrices to compute the eigenvalues later
        self.cov_matrices = None


    def step(self):
        """
        Performs one Monte Carlo step 
        We attempt to flip randomly selected nodes one by one.
        """

        # Loop N times so every node has the chance of being update per time step
        for _ in range(self.num_nodes):

            # Pick a random node
            node_idx = np.random.randint(0, self.num_nodes)
            
            # Identify the current state of that node
            s = self.current_state[node_idx]
            
            # The influence from neighbors
            # Row of Adjacency matrix @ State of all the noodes
            neighbor_sum = np.dot(self.adj_matrix[node_idx], self.current_state)
            
            # The total force applied on this node as per the Ising model
            # Force = (Coupling * Neighbors) + External field
            total_force = (self.J * neighbor_sum) + self.H
            
            # The Energy cost to flip the current node
            delta_E = 2 * s * total_force
            
            # The metropolis decision
            # if the energy decreases then we flip the node immediately
            # if the energy increases then we still have a prob of flipping due to random noise
            if delta_E < 0:

                self.current_state[node_idx] *= -1

            # Flip with probability P = exp(-dE/T)
            elif np.random.random() < np.exp(-delta_E / self.T):
                
                self.current_state[node_idx] *= -1



    def run(self, delta_H=None):
        """
        Runs the model for the specified length of time in the init
        If delta_H is specified, the stress moves up or down at each step. The first value of delta_H is ignored.
        """
        # Save the original H so we can reset it later
        original_H = self.H
        if delta_H is not None:
            if type(delta_H) == np.ndarray:
                if len(delta_H) != self.length:
                    raise ValueError(f"delta_H must be a float or an array with the same length as the simulation length ({self.length})")
            elif type(delta_H) == float:
                delta_H = np.full(self.length, delta_H)
            else:
                raise ValueError(f"delta_H must be a float or an array with the same length as the simulation length ({self.length})")
        
        # Warmup - Settling down from pure random noise
        # Run the physics, but we dont record the data
        for i in range(self.warmup):
            self.step()
            self.warmup_states[i] = self.current_state.copy()
        
        
        # Record the state right after warmup as t=0
        # copy() is necessary so we dont save a reference to the changing array of spins
        self.history[0] = self.current_state.copy()
        self.order_history[0] = self.get_order()
        
        #storing the external field/stress for use in later analysis
        self.h_history[0] = self.H
        
        for t in range(1, self.length):
            # change stress at each time step if requested
            if delta_H is not None:
                self.H += delta_H[t]
            
            # evolve the system one step
            self.step()
            
            # save to the pre-allocated arrays
            self.history[t] = self.current_state.copy()
            self.order_history[t] = self.get_order()
            self.h_history[t] = self.H
        
        # Reset H so the value stays consistent
        self.H = original_H


    def get_order(self):
        """
        Calculates the order parameter, that is, the average mood
        -1 is total depression and +1 is perfect health
        """
        return np.mean(self.current_state)
    
    def order(self):
        """
        Returns the timeline of the variation of the order parameter for each state of the network
        """
        return self.order_history


    def covariance_matrices(self, window=5):
        """
        Get a covariance matrix at each time step using the specified window size.
        The window uses warmup states for early timesteps, then transitions to history states.

        Returns:
            numpy array of shape (length, num_nodes, num_nodes) containing covariance matrices
        """
        if window > self.warmup + self.length:
            raise ValueError(f"Window size ({window}) cannot exceed warmup + length ({self.warmup + self.length})")

        all_states = np.vstack([self.warmup_states, self.history])
        cov_matrices = np.zeros((self.length, self.num_nodes, self.num_nodes))

        for t in range(self.length):
            current_idx = self.warmup + t
            start_idx = max(0, current_idx - window + 1)
            end_idx = current_idx + 1

            window_states = all_states[start_idx:end_idx]
            
            if window_states.shape[0] < 2:
                continue

            cov_matrices[t] = np.cov(window_states.T)

        # Save the covariance matrices for later use
        self.cov_matrices = cov_matrices
        return cov_matrices

    def sorted_cov_eigens(self):
        """
        Calculates the sorted eigenvalues and eigenvectors of the covariance matrix at each time step.
        """
        if self.cov_matrices is None:
            print("Covariance matrices have not been calculated. Calculating now...")
            self.covariance_matrices()
        
        all_eigenvalues = np.zeros((self.length, self.num_nodes))
        all_eigenvectors = np.zeros((self.length, self.num_nodes, self.num_nodes))
        
        for t, cov in enumerate(self.cov_matrices):
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            
            all_eigenvalues[t] = eigenvalues[sorted_indices]
            all_eigenvectors[t] = eigenvectors[:, sorted_indices]
        
        return all_eigenvalues, all_eigenvectors
    
    def get_early_warning_signal(self):
        """
        Stores the leading eigenvalue for each time step
        This is the specific method Chen et al. use as the warning signal
        
        Returns:
            numpy array
        """
        # Get all eigenvalues (sorted descending)
        eigenvalues, _ = self.sorted_cov_eigens()
        
        # Take the first column (the largest eigenvalue at each step)
        # This represents the variance along the system's most unstable dimension
        max_eigenvalues = eigenvalues[:, 0]
        
        return max_eigenvalues