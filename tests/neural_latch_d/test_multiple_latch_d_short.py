import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_latch_d import MultipleNeuralLatchD
from sPyBlocks.neural_not import MultipleNeuralNot
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 14.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    data_up_times_1 = [1.0, 3.0, 4.0, 6.0, 7.0, 9.0]
    data_up_times_2 = [3.0, 5.0, 7.0]
    signal_times = [1.0, 2.0, 3.0, 6.0, 8.0]
    data_sources = [sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times_1)),
                    sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times_2))]
    signal_source = sim.Population(1, sim.SpikeSourceArray(spike_times=signal_times))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    latches = MultipleNeuralLatchD(6, sim, global_params, neuron_params, std_conn, and_type="classic", include_not=False)
    not_gates = MultipleNeuralNot(2, sim, global_params, neuron_params, std_conn)
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    not_gates.connect_excitation([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])
    latches.connect_constant_spikes(constant_spike_source.latch.output_neuron)

    delayed_conn = sim.StaticSynapse(weight=std_conn.weight, delay=2 * std_conn.delay)

    not_gates.connect_inputs(data_sources, ini_pop_indexes=[[0], [1]])

    latches.connect_not_data(not_gates.not_array[0].output_neuron, component_indexes=[0, 1, 2])
    latches.connect_data(data_sources[0], conn=delayed_conn, component_indexes=[0, 1, 2])

    latches.connect_not_data(not_gates.not_array[1].output_neuron, component_indexes=[3, 4, 5])
    latches.connect_data(data_sources[1], conn=delayed_conn, component_indexes=[3, 4, 5])

    latches.connect_signals(signal_source, conn=delayed_conn)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for not_gate in not_gates.not_array:
        not_gate.output_neuron.record(('spikes'))

    for latch in latches.latch_array:
        latch.latch_sr.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    not_spikes = []
    for not_gate in not_gates.not_array:
        not_spikes.append(not_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    out_spikes = []
    for latch in latches.latch_array:
        out_spikes.append(latch.latch_sr.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_latch_d_short", simtime)

    trace.printSpikes(1, "Data signal 1", data_up_times_1, "#FFC000")
    trace.printSpikes(2, "Data signal 2", data_up_times_2, "#FFC000")
    trace.printSpikes(3, "Store signal", signal_times, "#92D050")
    trace.printSpikes(4, "NOT response (gate 0)", not_spikes[0], "#FF0000")
    trace.printSpikes(5, "NOT response (gate 1)", not_spikes[1], "#FF0000")
    for i in range(len(latches.latch_array)):
        trace.printSpikes(6 + i, "Latch " + str(i) + " spikes", out_spikes[i], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Latch): " + str(latches.latch_array[0].total_neurons) +
          "\nNumber of total connections (Latch): " + str(latches.latch_array[0].total_input_connections +
                                                          latches.latch_array[0].total_internal_connections +
                                                          latches.latch_array[0].total_output_connections))

    print(out_spikes)
