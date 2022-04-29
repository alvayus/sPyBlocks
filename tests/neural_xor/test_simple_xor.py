import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from neural_logic_gates.connection_functions import create_connections
from neural_logic_gates.neural_xor import NeuralXor
from neural_logic_gates.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 10.0  # (ms)

    # Other parameters
    n_inputs = 4
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times_3 = np.arange(1.0, simtime, 4.0)
    spike_times_2 = np.arange(1.0, simtime, 3.0)
    spike_times_1 = np.arange(1.0, simtime, 2.0)
    spike_times_0 = np.arange(1.0, simtime, 1.0)

    spike_source_3 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_3))
    spike_source_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_2))
    spike_source_1 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_1))
    spike_source_0 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_0))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    xor_gate = NeuralXor(n_inputs, sim, global_params, neuron_params, std_conn)

    # Testing
    xor_gate.connect_inputs(spike_source_3, end_pop_indexes=[3])
    xor_gate.connect_inputs(spike_source_2, end_pop_indexes=[2])
    xor_gate.connect_inputs(spike_source_1, end_pop_indexes=[1])
    xor_gate.connect_inputs(spike_source_0, end_pop_indexes=[0])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    xor_gate.input_neurons.record(('spikes', 'v'))
    xor_gate.x_neurons.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    in_spikes = xor_gate.input_neurons.get_data(variables=["spikes"]).segments[0].spiketrains
    in_voltage = xor_gate.input_neurons.get_data(variables=["v"]).segments[0].analogsignals[0]
    x_spikes = xor_gate.x_neurons.get_data(variables=["spikes"]).segments[0].spiketrains
    x_voltage = xor_gate.x_neurons.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_xor", simtime)

    trace.printSpikes(1, "Input signal 0", spike_times_0, "#92D050")
    trace.printSpikes(2, "Input signal 1", spike_times_1, "#92D050")
    trace.printSpikes(3, "Input signal 2", spike_times_2, "#92D050")
    trace.printSpikes(4, "Input signal 3", spike_times_3, "#92D050")

    for i in range(n_inputs):
        trace.printSpikes(5 + i * 2, 'Input neuron (neuron ' + str(i) + ')', in_spikes[i], "#FFF2CC")
        trace.printSpikes(6 + i * 2, 'X neuron (neuron ' + str(i) + ')', x_spikes[i], "#FFF2CC")

    values = np.zeros(int(simtime))
    for i in range(len(values)):
        for spike_array in x_spikes:
            if len(spike_array) > 0 and i in spike_array:
                values[i] = 1

    trace.printRow(7 + (n_inputs - 1) * 2, "OUTPUT", values, "#FFC000")

    trace.closeExcel()

    # Results
    print("Number of total neurons (XOR gate): " + str(xor_gate.total_neurons) +
          "\nNumber of total input connections (XOR gate): " + str(xor_gate.total_input_connections) +
          "\nNumber of total internal connections (XOR gate): " + str(xor_gate.total_internal_connections) +
          "\nNumber of total output connections (XOR gate): " + str(xor_gate.total_output_connections))

    print(in_spikes)
    print(x_spikes)

    '''plt.subplot(121)
    plt.plot(in_voltage.times, in_voltage)
    for i in range(0, n_inputs):
        plt.plot(in_spikes[i], [neuron_params["v_rest"]] * len(in_spikes[i]), 'ro', markersize=5)
        plt.vlines(in_spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('Input neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    in_label = mpatches.Patch(color='red', label='Input neuron spike')
    plt.legend(handles=[in_label])

    plt.subplot(122)
    plt.plot(x_voltage.times, x_voltage)
    for i in range(0, n_inputs):
        plt.plot(x_spikes[i], [neuron_params["v_rest"]] * len(x_spikes[i]), 'ro', markersize=5)
        plt.vlines(x_spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('X neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    x_label = mpatches.Patch(color='red', label='X neuron spike')
    plt.legend(handles=[x_label])

    plt.show()'''
