import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from neural_logic_gates.constant_spike_source import ConstantSpikeSource
from neural_logic_gates.neural_and import MultipleNeuralAnd
from neural_logic_gates.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 10.0  # (ms)

    # Other parameters
    show_graph = False
    n_inputs = 3
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times_1 = np.arange(1.0, simtime, global_params["min_delay"])
    spike_sources_1 = sim.Population(2, sim.SpikeSourceArray(spike_times=spike_times_1))
    spike_times_2 = [2.0, 6.0]
    spike_source_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_2))

    spike_sources_array = []
    for i in range(2):
        spike_sources_array.append(sim.PopulationView(spike_sources_1, [i]))
    spike_sources_array.append(spike_source_2)

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    classic_and_gates = MultipleNeuralAnd(3, n_inputs, sim, global_params, neuron_params, std_conn, build_type="classic")
    fast_and_gates = MultipleNeuralAnd(3, n_inputs, sim, global_params, neuron_params, std_conn, build_type="fast")
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    #classic_and_gates.connect_inputs(sim.Assembly(*[spike_sources_1, spike_source_2]), component_indexes=[0])
    classic_and_gates.connect_inputs(spike_sources_array, component_indexes=[0])
    classic_and_gates.connect_inputs([sim.PopulationView(spike_sources_1, [1]), spike_source_2],
                                     ini_pop_indexes=[[0], [1]], component_indexes=[1, 2])
    fast_and_gates.connect_inputs(spike_sources_array, component_indexes=[0])
    fast_and_gates.connect_inputs(spike_sources_array, ini_pop_indexes=[[1], [2]], component_indexes=[1, 2])
    fast_and_gates.connect_inhibition([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    constant_spike_source.latch.output_neuron.record(('spikes', 'v'))

    for gate in classic_and_gates.and_array:
        gate.output_neuron.record(('spikes', 'v'))
    for gate in fast_and_gates.and_array:
        gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    css_set_spike = [1]
    css_ff_spikes = constant_spike_source.latch.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains

    and_spikes = []
    and_voltage = []
    for gate in classic_and_gates.and_array:
        and_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        and_voltage.append(gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0])
    for gate in fast_and_gates.and_array:
        and_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        and_voltage.append(gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0])

    # End simulation
    sim.end()

    # Excel file creation
    n_classic = len(classic_and_gates.and_array)

    trace = SpikeTrace("test_multiple_and", simtime)

    trace.printSpikes(1, "Input signal 2", spike_times_1, "#92D050")
    trace.printSpikes(2, "Input signal 1", spike_times_1, "#92D050")
    trace.printSpikes(3, "Input signal 0", spike_times_2, "#92D050")
    trace.printSpikes(4, "Constant spike source (Set)", css_set_spike, "#FF0000")
    trace.printSpikes(5, "Constant spike source (FF)", css_ff_spikes[0], "#FF0000")

    for i in range(n_classic):
        trace.printSpikes(6 + i, 'Output neuron response (Classic AND ' + str(i) + ')', and_spikes[i][0], "#FFF2CC")
    for i in range(n_classic, n_classic + len(fast_and_gates.and_array)):
        trace.printSpikes(6 + i, 'Output neuron response (Fast AND ' + str(i-n_classic) + ')',
                          and_spikes[i][0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Classic AND gates): " + str(classic_and_gates.total_neurons) +
          "\nNumber of total input connections (Classic AND gates): " + str(classic_and_gates.total_input_connections) +
          "\nNumber of total internal connections (Classic AND gates): " + str(classic_and_gates.total_internal_connections) +
          "\nNumber of total output connections (Classic AND gates): " + str(classic_and_gates.total_output_connections))

    print("------------------------------------------")

    print("Number of total neurons (Fast AND gates): " + str(fast_and_gates.total_neurons) +
          "\nNumber of total input connections (Fast AND gates): " + str(fast_and_gates.total_input_connections) +
          "\nNumber of total internal connections (Fast AND gates): " + str(fast_and_gates.total_internal_connections) +
          "\nNumber of total output connections (Fast AND gates): " + str(fast_and_gates.total_output_connections))

    for i in range(n_classic):
        print(and_spikes[i])

        if show_graph:
            plt.plot(and_voltage[i].times, and_voltage[i])
            plt.plot(and_spikes[i], [neuron_params["v_rest"]] * len(and_spikes[i]), 'ro', markersize=5)
            plt.vlines(and_spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
            plt.title('Output neuron response (Classic AND ' + str(i) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            or_label = mpatches.Patch(color='red', label='AND neuron spike')
            plt.legend(handles=[or_label])
            plt.show()

    for i in range(n_classic, n_classic + len(fast_and_gates.and_array)):
        print(and_spikes[i])

        if show_graph:
            plt.plot(and_voltage[i].times, and_voltage[i])
            plt.plot(and_spikes[i], [neuron_params["v_rest"]] * len(and_spikes[i]), 'ro', markersize=5)
            plt.vlines(and_spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
            plt.title('Output neuron response (Fast AND ' + str(i) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            or_label = mpatches.Patch(color='red', label='AND neuron spike')
            plt.legend(handles=[or_label])
            plt.show()