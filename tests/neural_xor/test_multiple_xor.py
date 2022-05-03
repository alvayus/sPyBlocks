import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.neural_xor import MultipleNeuralXor
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 10.0  # (ms)

    # Other parameters
    show_graph = True
    n_components = 2
    n_inputs = 3
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times = [5.0]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    xor_gates = MultipleNeuralXor(n_components, n_inputs, sim, global_params, neuron_params, std_conn)

    # Testing
    tests = [[0, 0, 1], [1, 0, 1]]
    for i in range(len(tests)):
        one_indexes = np.nonzero(tests[i])[0]
        xor_gates.connect_inputs(spike_source, end_pop_indexes=one_indexes, component_indexes=[i])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in xor_gates.xor_array:
        gate.input_neurons.record(('spikes'))
        gate.x_neurons.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    total_input_spikes = []
    total_x_spikes = []
    spikes_per_gate = []
    voltage = []
    for gate in xor_gates.xor_array:
        input_spikes = gate.input_neurons.get_data(variables=["spikes"]).segments[0].spiketrains
        x_spikes = gate.x_neurons.get_data(variables=["spikes"]).segments[0].spiketrains

        total_input_spikes.append(input_spikes)
        total_x_spikes.append(x_spikes)
        spikes_per_gate.append(np.array(np.concatenate(x_spikes)))

        voltage.append(gate.x_neurons.get_data(variables=["v"]).segments[0].analogsignals[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_xor", simtime)

    trace.printSpikes(1, "Input signal", spike_times, "#92D050")

    for i in range(n_components):
        for j in range(n_inputs):
            trace.printSpikes(2 + i * n_inputs * 2 + j * 2 + i, 'Input neuron ' + str(j) + ' gate ' + str(i), total_input_spikes[i][j],
                              "#FFF2CC")
            trace.printSpikes(2 + i * n_inputs * 2 + j * 2 + 1 + i, 'X neuron ' + str(j) + ' gate ' + str(i), total_x_spikes[i][j],
                              "#FFF2CC")

        values = np.zeros(int(simtime))
        for k in range(len(values)):
            if len(total_x_spikes[i]) > 0 and k in total_x_spikes[i]:
                values[k] = 1

        trace.printRow(2 + (i + 1) * n_inputs * 2 + i, "OUTPUT gate " + str(i), values, "#FFC000")

    trace.closeExcel()

    # Results
    print("Number of total neurons (XOR gates): " + str(xor_gates.total_neurons) +
          "\nNumber of total input connections (XOR gates): " + str(xor_gates.total_input_connections) +
          "\nNumber of total internal connections (XOR gates): " + str(xor_gates.total_internal_connections) +
          "\nNumber of total output connections (XOR gates): " + str(xor_gates.total_output_connections))

    for i in range(len(xor_gates.xor_array)):
        print(spikes_per_gate[i])

        if show_graph:
            plt.plot(voltage[i].times, voltage[i])
            plt.plot(spikes_per_gate[i], [neuron_params["v_rest"]] * len(spikes_per_gate[i]), 'ro', markersize=5)
            plt.vlines(spikes_per_gate[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
            plt.title('Output neuron response (gate ' + str(i) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            or_label = mpatches.Patch(color='red', label='Output neuron spike')
            plt.legend(handles=[or_label])
            plt.show()