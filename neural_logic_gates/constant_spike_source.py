from neural_logic_gates.connection_functions import create_connections
from neural_logic_gates.neural_latch_sr import NeuralLatchSR


class ConstantSpikeSource:
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
        self.set_source = sim.Population(1, sim.SpikeSourceArray(spike_times=[1.0]))
        self.latch = NeuralLatchSR(sim, global_params, neuron_params, std_conn)

        self.total_neurons += self.set_source.size + self.latch.total_neurons
        self.total_internal_connections += self.latch.total_internal_connections

        # Create the connections
        created_connections = self.latch.connect_set(self.set_source)
        self.total_internal_connections += created_connections

        # Total internal delay
        self.delay = 0

    def connect_outputs(self, output_population, conn=None, rcp_type="excitatory"):
        if conn is None:
            conn = self.std_conn

        created_connections = 0

        created_connections += create_connections(self.set_source, output_population, self.sim, conn, rcp_type=rcp_type)
        created_connections += self.latch.connect_output(output_population)

        self.total_output_connections += created_connections
        return created_connections

    def get_output_neuron(self):
        return [self.set_source, self.latch.get_output_neuron()]