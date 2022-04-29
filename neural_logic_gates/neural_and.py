from .connection_functions import create_connections, multiple_connect, flatten
from .neural_or import NeuralOr


class NeuralAnd:
    def __init__(self, n_inputs, sim, global_params, neuron_params, std_conn, build_type="classic"):
        # Storing parameters
        self.n_inputs = n_inputs
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        if build_type == "classic" or build_type == "fast":
            self.build_type = build_type
        else:
            raise ValueError("This build type is not implemented.")

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        if build_type == "classic":
            self.or_gate = NeuralOr(sim, global_params, neuron_params, std_conn)

            self.total_neurons += self.or_gate.total_neurons
            self.total_internal_connections += self.or_gate.total_internal_connections

        self.output_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                            initial_values={'v': neuron_params["v_rest"]})
        self.total_neurons += self.output_neuron.size

        # Custom synapses
        self.inh_synapse = sim.StaticSynapse(weight=n_inputs - 1, delay=global_params["min_delay"])

        # Create the connections
        if build_type == "classic":
            created_connections = create_connections(self.or_gate.output_neuron, self.output_neuron, sim,
                                                     self.inh_synapse, rcp_type="inhibitory")

            self.total_internal_connections += created_connections

        # Total internal delay
        if build_type == "classic":
            self.delay = self.or_gate.delay + self.inh_synapse.delay
        else:
            self.delay = 0

    def connect_inhibition(self, input_population, conn=None, conn_all=True, rcp_type="inhibitory",
                           ini_pop_indexes=None, end_pop_indexes=None):
        if self.build_type == "fast":
            if conn is None:
                conn = self.inh_synapse

            created_connections = create_connections(input_population, self.output_neuron, self.sim, conn,
                                                     rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

            self.total_input_connections += created_connections
            return created_connections
        else:
            raise TypeError("connect_inhibition function is not allowed for Classic AND gates")

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = 0
        if self.build_type == "classic":
            created_connections += self.or_gate.connect_inputs(input_population, conn, rcp_type=rcp_type,
                                                               ini_pop_indexes=ini_pop_indexes)

        delayed_conn = self.sim.StaticSynapse(weight=conn.weight, delay=conn.delay + self.delay)
        created_connections += create_connections(input_population, self.output_neuron, self.sim, delayed_conn,
                                                  rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(self.output_neuron, output_population, self.sim, conn,
                                                 rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_inhibited_neuron(self):
        if self.build_type == "classic":
            return []
        else:
            return self.output_neuron

    def get_input_neurons(self):
        if self.build_type == "classic":
            return [self.or_gate.output_neuron, self.output_neuron]
        else:
            return self.output_neuron

    def get_output_neuron(self):
        return self.output_neuron


class MultipleNeuralAnd:
    def __init__(self, n_components, n_inputs, sim, global_params, neuron_params, std_conn, build_type="classic"):
        # Storing parameters
        self.n_components = n_components
        self.n_inputs = n_inputs
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn
        if build_type == "classic" or build_type == "fast":
            self.build_type = build_type
        else:
            raise ValueError("This type of AND gate is not implemented.")

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the array of multiple src
        self.and_array = []
        for i in range(n_components):
            and_gate = NeuralAnd(n_inputs, sim, global_params, neuron_params, std_conn, build_type=build_type)
            self.and_array.append(and_gate)

            self.total_neurons += and_gate.total_neurons
            self.total_internal_connections += and_gate.total_internal_connections

        # Custom synapses
        self.inh_synapse = self.and_array[0].inh_synapse

        # Total internal delay
        self.delay = self.and_array[0].delay

    def connect_inhibition(self, input_population, conn=None, conn_all=True, rcp_type="inhibitory",
                           ini_pop_indexes=None, end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.inh_synapse

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_inhibition", input_population, self.and_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_inputs", input_population, self.and_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_outputs", output_population, self.and_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_inhibited_neurons(self, flat=False):
        inhibited_neurons = []

        for i in range(self.n_components):
            inhibited_neurons.append(self.and_array[i].get_inhibited_neuron())

        if flat:
            return flatten(inhibited_neurons)
        else:
            return inhibited_neurons

    def get_input_neurons(self, flat=False):
        input_neurons = []

        for i in range(self.n_components):
            input_neurons.append(self.and_array[i].get_input_neurons())

        if flat:
            return flatten(input_neurons)
        else:
            return input_neurons

    def get_output_neurons(self):
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.and_array[i].get_output_neuron())

        return output_neurons
