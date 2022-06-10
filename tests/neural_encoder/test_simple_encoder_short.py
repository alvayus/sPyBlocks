from math import log2, ceil

import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_decoder import NeuralDecoder
from sPyBlocks.neural_encoder import NeuralEncoder
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 27.0  # (ms)

    # Other parameters
    n_inputs = 3
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times_0 = np.concatenate((np.arange(4.0, 7.0, global_params["min_delay"]),
                                    np.arange(10.0, 13.0, global_params["min_delay"]),
                                    np.arange(16.0, 19.0, global_params["min_delay"]),
                                    np.arange(22.0, 25.0, global_params["min_delay"])))
    spike_times_1 = np.concatenate((np.arange(7.0, 13.0, global_params["min_delay"]),
                                    np.arange(19.0, 25.0, global_params["min_delay"])))
    spike_times_2 = np.arange(13.0, 25.0, global_params["min_delay"])

    spike_source_0 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_0))
    spike_source_1 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_1))
    spike_source_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_2))

    spike_sources = [spike_source_0, spike_source_1, spike_source_2]

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    decoder = NeuralDecoder(n_inputs, sim, global_params, neuron_params, std_conn)
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)
    encoder = NeuralEncoder(2 ** n_inputs, sim, global_params, neuron_params, std_conn)

    # Testing
    decoder.connect_constant_spikes([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])
    decoder.connect_inputs(spike_sources, ini_pop_indexes=[[0], [1], [2]])

    input_population = [decoder.and_gates.and_array[i].output_neuron for i in range(decoder.n_outputs)]

    pop_len = len(input_population)
    input_indexes = range(pop_len)
    channel_indexes = range(encoder.n_inputs)
    if len(input_indexes) != len(channel_indexes):
        raise ValueError("There is not the same number of elements in input_indexes and channel_indexes")
    for i in range(pop_len):
        i_bin = format(channel_indexes[i], "0" + str(n_inputs) + 'b')
        i_bin_splitted = [j for j in reversed(i_bin)]
        connections = [k for k in range(0, len(i_bin_splitted)) if i_bin_splitted[k] == '1']
        encoder.connect_inputs(input_population, ini_pop_indexes=[input_indexes[i]], or_indexes=connections)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in decoder.and_gates.and_array:
        gate.output_neuron.record(('spikes'))

    for gate in encoder.or_gates.or_array:
        gate.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    decoder_spikes = []
    for gate in decoder.and_gates.and_array:
        decoder_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    encoder_spikes = []
    for gate in encoder.or_gates.or_array:
        encoder_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_encoder_short", simtime)

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

    and_length = len(decoder.and_gates.and_array)
    or_length = len(encoder.or_gates.or_array)

    for i in range(and_length):
        trace.printSpikes(4 + and_length - i, "Decoder response (AND gate " + str(i) + ")", decoder_spikes[i],
                          "#FFF2CC")

    for i in range(or_length):
        trace.printSpikes(4 + and_length + or_length - i, "Encoder response (OR gate " + str(i) + ")",
                          encoder_spikes[i], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Decoder + CSS): " + str(decoder.total_neurons +
                                                            constant_spike_source.total_neurons) +
          "\nNumber of total connections (Decoder + CSS): " + str(decoder.total_input_connections +
                                                                  decoder.total_internal_connections +
                                                                  decoder.total_output_connections +
                                                                  constant_spike_source.total_internal_connections))

    if decoder.and_type == "classic":
        neurons = 2 ** (n_inputs + 1) + n_inputs + 2
        connections = 2 ** n_inputs * (2 * n_inputs + 1) + 3 * n_inputs + 2
    else:
        neurons = 2 ** n_inputs + n_inputs + 2
        connections = 2 ** n_inputs * (n_inputs + 2) + 3 * n_inputs + 2

    print("Number of expected total neurons (Decoder + CSS): " + str(neurons))
    print("Number of expected total connections (Decoder + CSS): " + str(connections))

    print("------------------------------------------")

    print("Number of total neurons (Encoder): " + str(encoder.total_neurons) +
          "\nNumber of total connections (Encoder): " + str(encoder.total_input_connections +
                                                            encoder.total_internal_connections +
                                                            encoder.total_output_connections))

    neurons = ceil(log2(2 ** n_inputs))
    connections = sum(bin(x - 1).count("1") for x in range(2, 1 + 2 ** n_inputs))

    print("Number of expected total neurons (Encoder): " + str(neurons))
    print("Number of expected total connections (Encoder): " + str(connections))

    print(decoder_spikes)
    print(encoder_spikes)

    # Set matplotlib parameters
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.rcParams['font.size'] = '18'

    plt.subplot(131)
    plt.plot(spike_times_0, [1] * len(spike_times_0), 'ro', markersize=5)
    plt.plot(spike_times_1, [2] * len(spike_times_1), 'co', markersize=5)
    plt.plot(spike_times_2, [3] * len(spike_times_2), 'yo', markersize=5)
    plt.xlim([0, simtime])
    plt.yticks([1, 2, 3])
    plt.title('Inputs')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input indexes')

    plt.subplot(132)
    for i in range(len(decoder_spikes)):
        spike_times = decoder_spikes[i]
        plt.plot(spike_times, [i] * len(spike_times), 'mo', markersize=5)
    plt.xlim([0, simtime])
    plt.yticks(range(2 ** n_inputs))
    plt.title('Decoder response')
    plt.xlabel('Time (ms)')
    plt.ylabel('AND neuron index')

    plt.subplot(133)
    for i in range(len(encoder_spikes)):
        spike_times = encoder_spikes[i]
        plt.plot(spike_times, [i] * len(spike_times), 'mo', markersize=5)
    plt.xlim([0, simtime])
    plt.yticks(range(n_inputs))
    plt.title('Encoder response')
    plt.xlabel('Time (ms)')
    plt.ylabel('OR neuron index')

    # plt.savefig("test_simple_encoder_short.png", bbox_inches='tight')
    plt.show()
