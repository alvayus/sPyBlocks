import numpy as np
import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_memory import NeuralMemory
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    show_detailed_trace = False
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    data_up_times_bit0 = [10.0, 50.0]
    data_up_times_bit1 = [25.0, 50.0]
    signal_times_0 = [10.0, 25.0, 30.0]
    signal_times_1 = [50.0]
    signal_times_2 = [50.0]

    data_sources = [sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times_bit0)),
                    sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times_bit1))]
    signal_sources = [sim.Population(1, sim.SpikeSourceArray(spike_times=signal_times_0)),
                      sim.Population(1, sim.SpikeSourceArray(spike_times=signal_times_1)),
                      sim.Population(1, sim.SpikeSourceArray(spike_times=signal_times_2))]

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    memory = NeuralMemory(7, 4, sim, global_params, neuron_params, std_conn, and_type="fast")  # 32 bits

    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    memory.connect_constant_spikes([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])
    memory.connect_data(data_sources, ini_pop_indexes=[[0], [1], [], []])
    memory.connect_signals(signal_sources, ini_pop_indexes=[[0], [1], [2]])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in memory.decoder.not_gates.not_array:
        gate.output_neuron.record(('spikes'))

    for gate in memory.decoder.and_gates.and_array:
        gate.output_neuron.record(('spikes'))

    for flip_flop in memory.latches.latch_array:
        flip_flop.not_gate.output_neuron.record(('spikes'))
        flip_flop.and_gates.and_array[0].output_neuron.record(('spikes'))
        flip_flop.and_gates.and_array[1].output_neuron.record(('spikes'))
        flip_flop.latch_sr.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    decoder_not_spikes = []
    for gate in memory.decoder.not_gates.not_array:
        decoder_not_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    decoder_and_spikes = []
    for gate in memory.decoder.and_gates.and_array:
        decoder_and_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    ff_not_spikes = []
    ff_and_spikes = []
    out_spikes = []
    for flip_flop in memory.latches.latch_array:
        ff_not_spikes.append(flip_flop.not_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        ff_and_spikes.append(
            flip_flop.and_gates.and_array[0].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        ff_and_spikes.append(
            flip_flop.and_gates.and_array[1].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])
        out_spikes.append(flip_flop.latch_sr.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    # End simulation
    sim.end()

    # Expected channel and channel values (from input signals and decoder outputs respectively)
    simtime_int = int(simtime)
    expected_channels = ["" for time in range(simtime_int)]
    channels = ["" for time in range(simtime_int)]

    for time in range(simtime_int):
        # Expected channel
        tmp_channel = 0
        if time in signal_times_0:
            tmp_channel += 1
        if time in signal_times_1:
            tmp_channel += 2
        if time in signal_times_2:
            tmp_channel += 4
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
    trace = SpikeTrace("test_simple_memory", simtime)

    trace.printSpikes(1, "Data bit 1", data_up_times_bit1, "#FFC000")
    trace.printSpikes(2, "Data bit 0", data_up_times_bit0, "#FFC000")
    trace.printSpikes(3, "Signal 2", signal_times_2, "#92D050")
    trace.printSpikes(4, "Signal 1", signal_times_1, "#92D050")
    trace.printSpikes(5, "Signal 0", signal_times_0, "#92D050")

    if show_detailed_trace:
        decoder_not_length = len(decoder_not_spikes)
        decoder_and_length = len(decoder_and_spikes)
        ff_length = memory.capacity

        for i in range(decoder_not_length):
            trace.printSpikes(5 + decoder_not_length - i, "NOT response (gate " + str(i) + ")", decoder_not_spikes[i],
                              "#FF0000")

        for i in range(decoder_and_length):
            trace.printSpikes(5 + decoder_not_length + decoder_and_length - i, "AND response (gate " + str(i) + ")",
                              decoder_and_spikes[i], "#FFF2CC")

        trace.printRow(6 + decoder_not_length + decoder_and_length, "Channel (Expected)", expected_channels, "#FFC000")
        trace.printRow(7 + decoder_not_length + decoder_and_length, "Channel (Decoder)", channels, "#FFC000")

        for i in range(ff_length):
            trace.printSpikes(7 + decoder_not_length + decoder_and_length + 4 * (ff_length - i) - 3, "FF" + str(i) +
                              " NOT response", ff_not_spikes[i], "#FF0000")
            trace.printSpikes(7 + decoder_not_length + decoder_and_length + 4 * (ff_length - i) - 2, "FF" + str(i) +
                              " AND1 response (Reset)", ff_and_spikes[i * 2 + 1], "#FFF2CC")
            trace.printSpikes(7 + decoder_not_length + decoder_and_length + 4 * (ff_length - i) - 1, "FF" + str(i) +
                              " AND0 response (Set)", ff_and_spikes[i * 2], "#FFF2CC")
            trace.printSpikes(7 + decoder_not_length + decoder_and_length + 4 * (ff_length - i), "FF" + str(i) + " spikes",
                              out_spikes[i], "#FFF2CC")

        for i in range(memory.n_dir):
            trace.printRow(7 + decoder_not_length + decoder_and_length + 4 * ff_length + memory.n_dir - i,
                           "Register " + str(i) + " value", hex_values[i], "#FFF2CC")
    else:
        trace.printRow(6, "Channel (Expected)", expected_channels, "#FFC000")
        trace.printRow(7, "Channel (Decoder)", channels, "#FFC000")

        for i in range(memory.n_dir):
            trace.printRow(7 + memory.n_dir - i,
                           "Register " + str(i) + " value", hex_values[i], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Memory): " + str(memory.total_neurons) +
          "\nNumber of total input connections (Memory): " + str(memory.total_input_connections) +
          "\nNumber of total internal connections (Memory): " + str(memory.total_internal_connections) +
          "\nNumber of total output connections (Memory): " + str(memory.total_output_connections))
