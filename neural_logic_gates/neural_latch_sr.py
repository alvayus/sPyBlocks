from neural_logic_gates.connection_functions import create_connections, multiple_connect


class NeuralLatchSR:
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

        # Create the connections
        created_connections = create_connections(self.output_neuron, self.output_neuron, sim, std_conn)
        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = 0

    def connect_set(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                    end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(input_population, self.output_neuron, self.sim, conn,
                                                 rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_reset(self, input_population, conn=None, conn_all=True, rcp_type="inhibitory", ini_pop_indexes=None,
                      end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(input_population, self.output_neuron, self.sim, conn,
                                                 rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_output(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(self.output_neuron, output_population, self.sim, conn,
                                                 rcp_type=rcp_type, end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_set_neuron(self):
        return self.output_neuron

    def get_reset_neuron(self):
        return self.output_neuron

    def get_output_neuron(self):
        return self.output_neuron


class MultipleNeuralLatchSR:
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
        self.latch_array = []
        for i in range(n_components):
            latch = NeuralLatchSR(sim, global_params, neuron_params, std_conn)
            self.latch_array.append(latch)

            self.total_neurons += latch.total_neurons
            self.total_internal_connections += latch.total_internal_connections

        # Total internal delay
        self.delay = self.latch_array[0].delay

    def connect_set(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                    end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_set", input_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_reset(self, input_population, conn=None, conn_all=True, rcp_type="inhibitory", ini_pop_indexes=None,
                      end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_reset", input_population, self.latch_array, conn,
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

        created_connections = multiple_connect("connect_outputs", output_population, self.latch_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_set_neurons(self):
        set_neurons = []

        for i in range(self.n_components):
            set_neurons.append(self.latch_array[i].get_set_neuron())

        return set_neurons

    def get_reset_neurons(self):
        reset_neurons = []

        for i in range(self.n_components):
            reset_neurons.append(self.latch_array[i].get_reset_neuron())

        return reset_neurons

    def get_output_neurons(self):
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.latch_array[i].get_output_neuron())

        return output_neurons
