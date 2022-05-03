import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_and import NeuralAnd
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 50.0  # (ms)

    # Other parameters
    n_inputs = 4
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times_1_2 = [1.0, 2.0, 4.0, 11.0, 21.0, 41.0, 42.0]
    spike_times_3 = [1.0, 3.0, 4.0, 11.0, 41.0, 42.0]
    spike_times_4 = [11.0, 4.0, 41.0, 42.0, 43.0]

    spike_source_1_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_1_2))
    spike_source_3 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_3))
    spike_source_4 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_4))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    weighted_conn = sim.StaticSynapse(weight=std_conn.weight*2, delay=global_params["min_delay"])

    classic_and_gate = NeuralAnd(n_inputs, sim, global_params, neuron_params, std_conn)

    fast_and_gate = NeuralAnd(n_inputs, sim, global_params, neuron_params, std_conn, "fast")
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    classic_and_gate.connect_inputs(spike_source_1_2, weighted_conn)
    classic_and_gate.connect_inputs([spike_source_3, spike_source_4])

    fast_and_gate.connect_inputs(spike_source_1_2, weighted_conn)
    fast_and_gate.connect_inputs([spike_source_3, spike_source_4])
    fast_and_gate.connect_inhibition([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    classic_and_gate.or_gate.output_neuron.record(('spikes', 'v'))
    classic_and_gate.output_neuron.record(('spikes', 'v'))

    constant_spike_source.set_source.record(('spikes'))
    constant_spike_source.latch.output_neuron.record(('spikes'))
    fast_and_gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    inh_spikes_classic = classic_and_gate.or_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    inh_voltage_classic = classic_and_gate.or_gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]
    and_spikes_classic = classic_and_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    and_voltage_classic = classic_and_gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    set_source_spike = constant_spike_source.set_source.get_data(variables=["spikes"]).segments[0].spiketrains
    ff_spikes = constant_spike_source.latch.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    and_spikes_fast = fast_and_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    and_voltage_fast = fast_and_gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Excel file creation
    total_inhibitory_spikes = np.sort(np.array(np.concatenate([set_source_spike[0], ff_spikes[0]])))

    trace = SpikeTrace("test_simple_and", simtime)

    trace.printSpikes(1, "Input signal 3", spike_times_1_2, "#92D050")
    trace.printSpikes(2, "Input signal 2", spike_times_1_2, "#92D050")
    trace.printSpikes(3, "Input signal 1", spike_times_3, "#92D050")
    trace.printSpikes(4, "Input signal 0", spike_times_4, "#92D050")
    trace.printSpikes(5, "OR response (Classic AND)", inh_spikes_classic[0], "#FF0000")
    trace.printSpikes(6, "AND response (Classic AND)", and_spikes_classic[0], "#FFF2CC")
    trace.printSpikes(7, "Inhibitory spikes (Fast AND)", total_inhibitory_spikes, "#FF0000")
    trace.printSpikes(8, "AND response (Fast AND)", and_spikes_fast[0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Classic AND gate): " + str(classic_and_gate.total_neurons) +
          "\nNumber of total input connections (Classic AND gate): " + str(classic_and_gate.total_input_connections) +
          "\nNumber of total internal connections (Classic AND gate): " + str(classic_and_gate.total_internal_connections) +
          "\nNumber of total output connections (Classic AND gate): " + str(classic_and_gate.total_output_connections))

    print("------------------------------------------")

    print("Number of total neurons (Fast AND gate): " + str(fast_and_gate.total_neurons) +
          "\nNumber of total input connections (Fast AND gate): " + str(fast_and_gate.total_input_connections) +
          "\nNumber of total internal connections (Fast AND gate): " + str(fast_and_gate.total_internal_connections) +
          "\nNumber of total output connections (Fast AND gate): " + str(fast_and_gate.total_output_connections))

    print(inh_spikes_classic)
    print(and_spikes_classic)

    print(and_spikes_fast)

    plt.figure(figsize=(5, 4))
    plt.plot(spike_times_1_2, [3] * len(spike_times_1_2), 'bo', markersize=5)
    plt.plot(spike_times_1_2, [2] * len(spike_times_1_2), 'bo', markersize=5)
    plt.plot(spike_times_3, [1] * len(spike_times_3), 'bo', markersize=5)
    plt.plot(spike_times_4, [0] * len(spike_times_4), 'bo', markersize=5)
    plt.title('Input spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input connection number')
    plt.yticks([0, 1, 2, 3])

    plt.show()

    # Set general font size
    plt.rcParams['font.size'] = '24'

    f, (ax1, ax2) = plt.subplots(1, 2, sharey="all")

    ax1.plot(inh_voltage_classic.times, inh_voltage_classic)
    ax1.plot(inh_spikes_classic, [neuron_params["v_rest"]] * len(inh_spikes_classic), 'ro', markersize=5)
    ax1.vlines(inh_spikes_classic, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    ax1.set_title('OR neuron response')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane potential (mV)')
    or_label = mpatches.Patch(color='red', label='OR neuron spike')
    ax1.legend(handles=[or_label])

    ax2.plot(and_voltage_classic.times, and_voltage_classic)
    ax2.plot(and_spikes_classic, [neuron_params["v_rest"]] * len(and_spikes_classic), 'ro', markersize=5)
    ax2.vlines(and_spikes_classic, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    ax2.set_title('Output neuron response')
    ax2.set_xlabel('Time (ms)')
    output_label = mpatches.Patch(color='red', label='Output neuron spike')
    ax2.legend(handles=[output_label])

    plt.show()

    plt.plot(and_voltage_fast.times, and_voltage_fast)
    plt.plot(and_spikes_fast, [neuron_params["v_rest"]] * len(and_spikes_fast), 'ro', markersize=5)
    plt.vlines(and_spikes_fast, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('Output neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    output_label = mpatches.Patch(color='red', label='Output neuron spike')
    plt.legend(handles=[output_label])

    plt.show()
