from math import ceil, log2

import numpy as np
import spynnaker8 as sim

from sPyBlocks.connection_functions import truth_table_column
from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_memory import NeuralMemory
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 50.0  # (ms)

    # Other parameters
    show_detailed_trace = False
    n_dir = 3
    n_signals = ceil(log2(n_dir + 1))
    n_bits = 3
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    dir_times = []
    data_times = []

    for i in range(n_signals):
        times = truth_table_column(ceil(simtime), i, select=1)
        dir_times.append(times)

    for i in range(n_bits):
        times = truth_table_column(ceil(simtime), i, select=1)
        data_times.append(times)

    dir_sources = []
    data_sources = []

    for i in range(n_signals):
        dir_sources.append(sim.Population(1, sim.SpikeSourceArray(spike_times=dir_times[i])))

    for i in range(n_bits):
        data_sources.append(sim.Population(1, sim.SpikeSourceArray(spike_times=data_times[i])))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    memory = NeuralMemory(n_dir, n_bits, sim, global_params, neuron_params, std_conn, and_type="fast")

    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    memory.connect_constant_spikes([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])
    memory.connect_signals(dir_sources, ini_pop_indexes=[[i] for i in range(n_signals)])
    memory.connect_data(data_sources, ini_pop_indexes=[[i] for i in range(n_bits)])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in memory.decoder.not_gates.not_array:
        gate.output_neuron.record(('spikes'))

    for gate in memory.decoder.and_gates.and_array:
        gate.output_neuron.record(('spikes'))

    for not_gate in memory.not_gates.not_array:
        not_gate.output_neuron.record(('spikes'))

    for latch in memory.latches.latch_array:
        latch.and_gates.and_array[0].output_neuron.record(('spikes'))
        latch.and_gates.and_array[1].output_neuron.record(('spikes'))
        latch.latch_sr.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    decoder_not_spikes = []
    for gate in memory.decoder.not_gates.not_array:
        decoder_not_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    decoder_and_spikes = []
    for gate in memory.decoder.and_gates.and_array:
        decoder_and_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    not_spikes = []
    for not_gate in memory.not_gates.not_array:
        not_spikes.append(not_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    latch_and_spikes = []
    out_spikes = []
    for latch in memory.latches.latch_array:
        latch_and_spikes.append(
            latch.and_gates.and_array[0].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        latch_and_spikes.append(
            latch.and_gates.and_array[1].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        out_spikes.append(latch.latch_sr.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    # End simulation
    sim.end()

    # Expected channel and channel values (from input signals and decoder outputs respectively)
    simtime_int = int(simtime)
    expected_channels = ["" for time in range(simtime_int)]
    channels = ["" for time in range(simtime_int)]

    for time in range(simtime_int):
        # Expected channel
        tmp_channel = 0
        for i in range(n_signals):
            if time in dir_times[i]:
                tmp_channel += 2 ** i
        if tmp_channel != 0:
            expected_channels[time] = str(tmp_channel)

        # Decoder output channel
        for i in range(1, len(decoder_and_spikes)):
            if time in decoder_and_spikes[i]:
                channels[time] = str(i)

    # Values calculation for each memory direction
    values = np.zeros((memory.n_dir, int(simtime)), dtype=int)
    hex_values = [["" for j in range(int(simtime))] for i in range(memory.n_dir)]
    for time in range(int(simtime)):
        for i in range(memory.n_dir):  # For each direction
            for j in range(memory.width):  # For each bit
                if time in out_spikes[i * memory.width + j]:
                    values[i][time] += 2 ** j

            if values[i][time] != 0:
                hex_values[i][time] = hex(values[i][time])

    # Excel file creation
    trace = SpikeTrace("test_memory_stress_1_mini", simtime)

    for i in range(n_bits):
        trace.printSpikes(n_bits - i, "Data bit " + str(i), data_times[i], "#FFC000")
    for i in range(n_signals):
        trace.printSpikes(n_bits + n_signals - i, "Signal " + str(i), dir_times[i], "#92D050")

    if show_detailed_trace:
        decoder_not_length = len(decoder_not_spikes)
        decoder_and_length = len(decoder_and_spikes)
        n_latches = memory.capacity

        for i in range(decoder_not_length):
            trace.printSpikes(n_bits + n_signals + decoder_not_length - i, "NOT response (gate " + str(i) + ")",
                              decoder_not_spikes[i], "#FF0000")

        for i in range(decoder_and_length):
            trace.printSpikes(n_bits + n_signals + decoder_not_length + decoder_and_length - i,
                              "AND response (gate " + str(i) + ")", decoder_and_spikes[i], "#FFF2CC")

        trace.printRow(n_bits + n_signals + decoder_not_length + decoder_and_length + 1, "Channel (Expected)",
                       expected_channels, "#FFC000")
        trace.printRow(n_bits + n_signals + decoder_not_length + decoder_and_length + 2, "Channel (Decoder)",
                       channels, "#FFC000")

        current_row = n_bits + n_signals + decoder_not_length + decoder_and_length + 2
        for i in range(n_bits):
            trace.printSpikes(current_row + n_bits - i, "External NOT" + str(i), not_spikes[i], "#FF0000")

        current_row = n_bits * 2 + n_signals + decoder_not_length + decoder_and_length + 2
        for i in range(n_latches):
            trace.printSpikes(current_row + 3 * (n_latches - i) - 2, "Latch" + str(i) +
                              " AND1 response (Reset)", latch_and_spikes[i * 2 + 1], "#FFF2CC")
            trace.printSpikes(current_row + 3 * (n_latches - i) - 1, "Latch" + str(i) +
                              " AND0 response (Set)", latch_and_spikes[i * 2], "#FFF2CC")
            trace.printSpikes(current_row + 3 * (n_latches - i), "Latch" + str(i) + " spikes",
                              out_spikes[i], "#FFF2CC")

        for i in range(memory.n_dir):
            trace.printRow(current_row + 3 * n_latches + memory.n_dir - i,
                           "Register " + str(i) + " value", hex_values[i], "#FFF2CC")
    else:
        trace.printRow(n_bits + n_signals + 1, "Channel (Expected)", expected_channels, "#FFC000")
        trace.printRow(n_bits + n_signals + 2, "Channel (Decoder)", channels, "#FFC000")

        for i in range(memory.n_dir):
            trace.printRow(n_bits + n_signals + 2 + memory.n_dir - i,
                           "Register " + str(i) + " value", hex_values[i], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Memory): " + str(memory.total_neurons) +
          "\nNumber of total input connections (Memory): " + str(memory.total_input_connections) +
          "\nNumber of total internal connections (Memory): " + str(memory.total_internal_connections) +
          "\nNumber of total output connections (Memory): " + str(memory.total_output_connections))
