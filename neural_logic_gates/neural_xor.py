import numpy as np

from neural_logic_gates.connection_functions import create_connections, multiple_connect


class NeuralXor:
    def __init__(self, n_inputs, sim, global_params, neuron_params, std_conn):
        # Storing parameters
        self.n_inputs = n_inputs
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
        self.input_neurons = sim.Population(n_inputs, sim.IF_curr_exp(**neuron_params),
                                            initial_values={'v': neuron_params["v_rest"]})
        self.x_neurons = sim.Population(n_inputs, sim.IF_curr_exp(**neuron_params),
                                        initial_values={'v': neuron_params["v_rest"]})

        self.total_neurons += self.input_neurons.size + self.x_neurons.size

        # Create the connections
        created_connections = 0

        created_connections += create_connections(self.input_neurons, self.x_neurons, sim, std_conn, False)

        input_indexes = range(n_inputs)
        for i in input_indexes:
            inh_indexes = np.delete(input_indexes, i)

            created_connections += create_connections(self.input_neurons, self.x_neurons, sim, std_conn,
                                                      rcp_type="inhibitory", ini_pop_indexes=[i],
                                                      end_pop_indexes=inh_indexes)

        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = self.std_conn.delay

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):

        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(input_population, self.input_neurons, self.sim, conn,
                                                 conn_all=conn_all,
                                                 rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                                 end_pop_indexes=end_pop_indexes)

        self.total_input_connections += created_connections
        return created_connections

    def connect_outputs(self, output_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                        end_pop_indexes=None):
        if conn is None:
            conn = self.std_conn

        created_connections = create_connections(self.x_neurons, output_population, self.sim, conn, conn_all=conn_all,
                                                 rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                                 end_pop_indexes=end_pop_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neurons(self):
        return self.input_neurons

    def get_output_neurons(self):
        return self.x_neurons


class MultipleNeuralXor:
    def __init__(self, n_components, n_inputs, sim, global_params, neuron_params, std_conn):
        # Storing parameters
        self.n_components = n_components
        self.n_inputs = n_inputs
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
        self.xor_array = []
        for i in range(n_components):
            xor_gate = NeuralXor(n_inputs, sim, global_params, neuron_params, std_conn)
            self.xor_array.append(xor_gate)

            self.total_neurons += xor_gate.total_neurons
            self.total_internal_connections += xor_gate.total_internal_connections

        # Total internal delay
        self.delay = self.xor_array[0].delay

    def connect_inputs(self, input_population, conn=None, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None, component_indexes=None):
        if conn is None:
            conn = self.std_conn

        if component_indexes is None:
            component_indexes = range(self.n_components)

        created_connections = multiple_connect("connect_inputs", input_population, self.xor_array, conn,
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

        created_connections = multiple_connect("connect_outputs", output_population, self.xor_array, conn,
                                               conn_all=conn_all, rcp_type=rcp_type, ini_pop_indexes=ini_pop_indexes,
                                               end_pop_indexes=end_pop_indexes, component_indexes=component_indexes)

        self.total_output_connections += created_connections
        return created_connections

    def get_input_neurons(self):
        input_neurons = []

        for i in range(self.n_components):
            input_neurons.append(self.xor_array[i].get_input_neurons())

        return input_neurons

    def get_output_neurons(self):
        output_neurons = []

        for i in range(self.n_components):
            output_neurons.append(self.xor_array[i].get_output_neurons())

        return output_neurons
