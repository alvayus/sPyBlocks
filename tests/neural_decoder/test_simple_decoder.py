import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_decoder import NeuralDecoder
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 200.0  # (ms)

    # Other parameters
    n_inputs = 3
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times_0 = np.concatenate((np.arange(10.0, 40.0, global_params["min_delay"]),
                                    np.arange(60.0, 90.0, global_params["min_delay"])))
    spike_times_1 = np.concatenate((np.arange(10.0, 40.0, global_params["min_delay"]),
                                    np.arange(40.0, 60.0, global_params["min_delay"])))
    spike_times_2 = np.concatenate((np.arange(10.0, 40.0, global_params["min_delay"]),
                                    np.arange(100.0, 130.0, global_params["min_delay"])))

    spike_source_0 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_0))
    spike_source_1 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_1))
    spike_source_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_2))

    spike_sources = [spike_source_0, spike_source_1, spike_source_2]

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    decoder = NeuralDecoder(n_inputs, sim, global_params, neuron_params, std_conn, and_type="classic")
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    decoder.connect_constant_spikes([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])

    decoder.connect_inputs(spike_sources, ini_pop_indexes=[[0], [1], [2]])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in decoder.not_gates.not_array:
        gate.output_neuron.record(('spikes'))

    for gate in decoder.and_gates.and_array:
        if decoder.and_type == "classic":
            gate.or_gate.output_neuron.record(('spikes'))
        gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    not_spikes = []
    for gate in decoder.not_gates.not_array:
        not_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    or_spikes = []
    out_spikes = []
    out_voltage = []
    for gate in decoder.and_gates.and_array:
        if decoder.and_type == "classic":
            or_spikes.append(gate.or_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        out_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        out_voltage.append(gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_decoder", simtime)

    trace.printSpikes(1, "Input signal 2", spike_times_2, "#8DB4E2")
    trace.printSpikes(2, "Input signal 1", spike_times_1, "#8DB4E2")
    trace.printSpikes(3, "Input signal 0", spike_times_0, "#8DB4E2")

    values = np.zeros(int(simtime))
    for i in range(len(values)):
        if i in spike_times_0:
            values[i] += 1
        if i in spike_times_1:
            values[i] += 2
        if i in spike_times_2:
            values[i] += 4

    trace.printRow(4, "CHANNEL", values, "#FFC000")

    not_length = len(decoder.not_gates.not_array)
    and_length = len(decoder.and_gates.and_array)

    for i in range(not_length):
        trace.printSpikes(4 + not_length - i, "NOT response (gate " + str(i) + ")", not_spikes[i], "#FF0000")

    for i in range(len(decoder.and_gates.and_array)):
        trace.printSpikes(4 + not_length + and_length - i, "AND response (gate " + str(i) + ")", out_spikes[i],
                          "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Decoder): " + str(decoder.total_neurons) +
          "\nNumber of total input connections (Decoder): " + str(decoder.total_input_connections) +
          "\nNumber of total internal connections (Decoder): " + str(decoder.total_internal_connections) +
          "\nNumber of total output connections (Decoder): " + str(decoder.total_output_connections))

    not_lengths = []
    for array in not_spikes:
        not_lengths.append(len(array))

    out_lengths = []
    for array in out_spikes:
        out_lengths.append(len(array))

    print(not_spikes)
    if decoder.and_type == "classic":
        print("\n")
        print(or_spikes)
    print("\n")
    print(out_spikes)

    plt.subplot(131)
    plt.plot(spike_times_0, [1] * len(spike_times_0), 'ro', markersize=5)
    plt.plot(spike_times_1, [2] * len(spike_times_1), 'co', markersize=5)
    plt.plot(spike_times_2, [3] * len(spike_times_2), 'yo', markersize=5)
    plt.yticks([1, 2, 3])
    plt.title('Inputs')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input indexes')

    plt.subplot(132)
    plt.bar(range(len(not_lengths)), not_lengths, color="tab:red")
    plt.title('NOT neuron responses')
    plt.xlabel('NOT neuron indexes')
    plt.ylabel('Number of spikes')
    plt.xticks(range(len(not_lengths)))

    plt.subplot(133)
    plt.bar(range(len(out_lengths)), out_lengths, color="tab:red")
    plt.title('AND neuron responses')
    plt.xlabel('AND neuron indexes')
    plt.ylabel('Number of spikes')
    plt.xticks(range(len(out_lengths)))

    plt.show()

    '''for i in range(len(decoder.and_gates.and_array)):
        plt.title("AND gate " + str(i))
        plt.plot(out_voltage[i].times, out_voltage[i])
        plt.show()'''
