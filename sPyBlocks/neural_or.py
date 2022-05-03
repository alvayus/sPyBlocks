from sPyBlocks.connection_functions import create_connections, multiple_connect


class NeuralOr:
    def __init__(self, sim, global_params, neuron_params, std_conn):
        # Storing parameters
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the neurons
        self.output_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params),
                                            initial_values={'v': neuron_params["v_rest"]})
        self.total_neurons += self.output_neuron.size

        # Total internal delay
        self.delay = 0

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(input_population, self.output_neuron, self.sim, conn, rcp_type=rcp_type,
                                                 ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(self.output_neuron, output_population, self.sim, conn, rcp_type=rcp_type,
                                                 end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neuron(self):
        return self.output_neuron

    def get_output_neuron(self):
        return self.output_neuron


class MultipleNeuralOr:
    def __init__(self, n_components, sim, global_params, neuron_params, std_conn):
        # Storing parameters
        self.n_components = n_components
        self.sim = sim
        self.global_params = global_params
        self.neuron_params = neuron_params
        self.std_conn = std_conn

        # Neuron and connection amounts
        self.total_neurons = 0
        self.total_input_connections = 0
        self.total_internal_connections = 0
        self.total_output_connections = 0

        # Create the array of multiple src
        self.or_array = []
        for i in range(n_components):
            or_gate = NeuralOr(sim, global_params, neuron_params, std_conn)
            self.or_array.append(or_gate)

            self.total_neurons += or_gate.total_neurons
            self.total_internal_connections += or_gate.total_internal_connections

        # Total internal delay
        self.delay = self.or_array[0].delay

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_inputs", input_population, self.or_array, conn,
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

        created_connections = multiple_connect("connect_outputs", output_population, self.or_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neurons(self):
        input_neurons = []

        for i in range(self.n_components):
            input_neurons.append(self.or_array[i].get_input_neuron())

        return input_neurons

    def get_output_neurons(self):
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.or_array[i].get_output_neuron())

        return output_neurons
