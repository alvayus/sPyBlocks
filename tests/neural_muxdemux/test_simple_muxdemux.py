import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_muxdemux import NeuralMuxDemux
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    n_select = 2
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    # spike_times_s0 = np.arange(21.0, 102.0, global_params["min_delay"])
    spike_times_s0 = np.concatenate((np.arange(10.0, 40.0, global_params["min_delay"]),
                                     np.arange(60.0, 90.0, global_params["min_delay"])))
    spike_times_s1 = np.concatenate((np.arange(10.0, 40.0, global_params["min_delay"]),
                                     np.arange(40.0, 60.0, global_params["min_delay"])))

    spike_times_x1 = np.arange(1.0, simtime, global_params["min_delay"])
    spike_times_x2 = np.arange(1.0, simtime, global_params["min_delay"] * 2)
    spike_times_x3 = np.arange(1.0, simtime, global_params["min_delay"] * 4)
    spike_times_x4 = np.arange(1.0, simtime, global_params["min_delay"] * 8)

    spike_sources_s0 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_s0))
    spike_sources_s1 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_s1))

    spike_sources_x1 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_x1))
    spike_sources_x2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_x2))
    spike_sources_x3 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_x3))
    spike_sources_x4 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_x4))

    spike_sources_array = [spike_sources_x1, spike_sources_x2, spike_sources_x3, spike_sources_x4]

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    multiplexer = NeuralMuxDemux(n_select, sim, global_params, neuron_params, std_conn, build_type="mux",
                                 and_type="classic")
    demultiplexer = NeuralMuxDemux(n_select, sim, global_params, neuron_params, std_conn, build_type="demux",
                                   and_type="classic")

    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    multiplexer.connect_constant_spikes(
        [constant_spike_source.set_source, constant_spike_source.latch.output_neuron])
    demultiplexer.connect_constant_spikes(
        [constant_spike_source.set_source, constant_spike_source.latch.output_neuron])

    multiplexer.connect_signals([spike_sources_s0, spike_sources_s1], ini_pop_indexes=[[0], [1]])
    demultiplexer.connect_signals([spike_sources_s0, spike_sources_s1], ini_pop_indexes=[[0], [1]])

    multiplexer.connect_inputs(spike_sources_array, ini_pop_indexes=[[0], [1], [2], [3]])  # OneToOne
    demultiplexer.connect_inputs(multiplexer.or_gate.output_neuron)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in multiplexer.not_gates.not_array:
        gate.output_neuron.record(('spikes'))

    for gate in multiplexer.and_gates.and_array:
        gate.output_neuron.record(('spikes'))

    multiplexer.or_gate.output_neuron.record(('spikes'))

    for gate in demultiplexer.not_gates.not_array:
        gate.output_neuron.record(('spikes'))

    for gate in demultiplexer.and_gates.and_array:
        gate.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    mux_not_spikes = []
    for gate in multiplexer.not_gates.not_array:
        mux_not_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    mux_and_spikes = []
    for gate in multiplexer.and_gates.and_array:
        mux_and_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    mux_output_spikes = multiplexer.or_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains

    demux_not_spikes = []
    for gate in multiplexer.not_gates.not_array:
        demux_not_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    demux_output_spikes = []
    for gate in multiplexer.and_gates.and_array:
        demux_output_spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_muxdemux", simtime)

    trace.printSpikes(1, "Select signal 1", spike_times_s1, "#8DB4E2")
    trace.printSpikes(2, "Select signal 0", spike_times_s0, "#8DB4E2")

    values = np.zeros(int(simtime))
    for i in range(len(values)):
        if i in spike_times_s0:
            values[i] += 1
        if i in spike_times_s1:
            values[i] += 2

    trace.printRow(3, "CHANNEL", values, "#FFC000")

    trace.printSpikes(4, "Input signal 3", spike_times_x4, "#92D050")
    trace.printSpikes(5, "Input signal 2", spike_times_x3, "#92D050")
    trace.printSpikes(6, "Input signal 1", spike_times_x2, "#92D050")
    trace.printSpikes(7, "Input signal 0", spike_times_x1, "#92D050")

    mux_not_length = len(multiplexer.not_gates.not_array)
    mux_and_length = len(multiplexer.and_gates.and_array)
    demux_not_length = len(demultiplexer.not_gates.not_array)
    demux_and_length = len(demultiplexer.and_gates.and_array)

    for i in range(mux_not_length):
        trace.printSpikes(7 + mux_not_length - i, "MUX NOT (gate " + str(i) + ")", mux_not_spikes[i][0], "#FF0000")

    for i in range(mux_and_length):
        trace.printSpikes(7 + mux_not_length + mux_and_length - i, "MUX AND (gate " + str(i) + ")",
                          mux_and_spikes[i][0], "#FFF2CC")

    trace.printSpikes(7 + mux_not_length + mux_and_length, "Multiplexer response", mux_output_spikes[0], "#FFF2CC")

    for i in range(demux_not_length):
        trace.printSpikes(7 + mux_not_length + mux_and_length + demux_not_length - i, "DEMUX NOT (gate " + str(i) + ")",
                          demux_not_spikes[i][0], "#FF0000")

    for i in range(demux_and_length):
        trace.printSpikes(7 + mux_not_length + mux_and_length + demux_not_length + demux_and_length - i,
                          "DEMUX AND (gate " + str(i) + ")", demux_output_spikes[i][0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Multiplexer + CSS): " + str(multiplexer.total_neurons +
                                                                constant_spike_source.total_neurons) +
          "\nNumber of total connections (Multiplexer + CSS): " + str(multiplexer.total_input_connections +
                                                                      multiplexer.total_internal_connections +
                                                                      multiplexer.total_output_connections +
                                                                      constant_spike_source.total_internal_connections))

    if multiplexer.and_type == "classic":
        neurons = 2 ** (n_select + 1) + n_select + 3
        connections = 2 ** n_select * (2 * n_select + 4) + 3 * n_select + 2
    else:
        neurons = 2 ** n_select + n_select + 3
        connections = 2 ** n_select * (n_select + 4) + 3 * n_select + 2

    print("Number of expected total neurons (Multiplexer + CSS): " + str(neurons))
    print("Number of expected total connections (Multiplexer + CSS): " + str(connections))

    print("------------------------------------------")

    print("Number of total neurons (Demultiplexer + CSS): " + str(demultiplexer.total_neurons +
                                                                  constant_spike_source.total_neurons) +
          "\nNumber of total connections (Demultiplexer + CSS): " + str(demultiplexer.total_input_connections +
                                                                        demultiplexer.total_internal_connections +
                                                                        demultiplexer.total_output_connections +
                                                                        constant_spike_source.total_internal_connections))

    if demultiplexer.and_type == "classic":
        neurons = 2 ** (n_select + 1) + n_select + 2
        connections = 2 ** n_select * (2 * n_select + 3) + 3 * n_select + 2
    else:
        neurons = 2 ** n_select + n_select + 2
        connections = 2 ** n_select * (n_select + 3) + 3 * n_select + 2

    print("Number of expected total neurons (Demultiplexer + CSS): " + str(neurons))
    print("Number of expected total connections (Demultiplexer + CSS): " + str(connections))

    print(mux_output_spikes)
    print(demux_output_spikes)

    # Set matplotlib parameters
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams['font.size'] = '14'

    plt.subplot(221)
    plt.plot(spike_times_s0, [0] * len(spike_times_s0), 'mo', markersize=5)
    plt.plot(spike_times_s1, [1] * len(spike_times_s1), 'mo', markersize=5)
    plt.xlim([0, simtime])
    plt.yticks([0, 1], labels=["S0", "S1"])
    plt.title('Selection signals')
    plt.xlabel('Time (ms)')
    plt.ylabel('Selection signal index')

    plt.subplot(222)
    plt.plot(spike_times_x1, [1] * len(spike_times_x1), 'ro', markersize=5)
    plt.plot(spike_times_x2, [2] * len(spike_times_x2), 'co', markersize=5)
    plt.plot(spike_times_x3, [3] * len(spike_times_x3), 'yo', markersize=5)
    plt.plot(spike_times_x4, [4] * len(spike_times_x4), 'go', markersize=5)
    plt.yticks([1, 2, 3, 4], labels=["D0", "D1", "D2", "D3"])
    plt.title('Data signals')
    plt.xlabel('Time (ms)')
    plt.ylabel('Data signal index')

    plt.subplot(223)
    plt.plot(mux_output_spikes, [0] * len(mux_output_spikes), 'mo', markersize=5)
    plt.xlim([0, simtime])
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        labelleft=False)
    plt.yticks([0])
    plt.title('Multiplexer response')
    plt.xlabel('Time (ms)')

    plt.subplot(224)
    for i in range(len(demux_output_spikes)):
        spike_times = demux_output_spikes[i]
        plt.plot(spike_times, [i] * len(spike_times), 'mo', markersize=5)
    plt.xlim([0, simtime])
    plt.yticks([0, 1, 2, 3], labels=["D0", "D1", "D2", "D3"])
    plt.title('Demultiplexer response')
    plt.xlabel('Time (ms)')
    plt.ylabel('AND neuron index')

    plt.tight_layout()
    #plt.savefig("test_simple_muxdemux.png", bbox_inches='tight')
    plt.show()
