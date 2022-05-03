import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_not import MultipleNeuralNot
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    show_graph = False
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times = [1.0, 51.0, 75.0]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    not_gates = MultipleNeuralNot(6, sim, global_params, neuron_params, std_conn)
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    test = [1, 0, 1, 0, 0, 1]
    connect_indexes = np.nonzero(test)[0]
    not_gates.connect_inputs(spike_source, component_indexes=connect_indexes)
    not_gates.connect_excitation([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    constant_spike_source.latch.output_neuron.record(('spikes', 'v'))

    for gate in not_gates.not_array:
        gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    css_set_spike = [1]
    css_ff_spikes = constant_spike_source.latch.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains

    spikes = []
    voltage = []
    for gate in not_gates.not_array:
        spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        voltage.append(gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_not", simtime)

    trace.printSpikes(1, "Input signal", spike_times, "#92D050")
    trace.printSpikes(2, "Constant spike source (Set)", css_set_spike, "#FF0000")
    trace.printSpikes(3, "Constant spike source (FF)", css_ff_spikes[0], "#FF0000")
    for i in range(len(not_gates.not_array)):
        trace.printSpikes(4 + i, "NOT response (gate " + str(i) + ")", spikes[i][0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (NOT gates): " + str(not_gates.total_neurons) +
          "\nNumber of total input connections (NOT gates): " + str(not_gates.total_input_connections) +
          "\nNumber of total internal connections (NOT gates): " + str(not_gates.total_internal_connections) +
          "\nNumber of total output connections (NOT gates): " + str(not_gates.total_output_connections))

    if show_graph:
        for i in range(len(not_gates.not_array)):
            print(spikes[i])

            plt.plot(voltage[i].times, voltage[i])
            plt.plot(spikes[i], [neuron_params["v_rest"]] * len(spikes[i]), 'ro', markersize=5)
            plt.vlines(spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
            plt.title('Output neuron response (gate ' + str(i) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            or_label = mpatches.Patch(color='red', label='Output neuron spike')
            plt.legend(handles=[or_label])
            plt.show()