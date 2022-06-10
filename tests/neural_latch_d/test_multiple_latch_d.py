import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_latch_d import MultipleNeuralLatchD
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    data_up_times_1 = [10.0, 20.0, 45.0, 70.0, 72.0]
    data_up_times_2 = [15.0, 25.0, 45.0, 69.0, 71.0]
    signal_times = [10.0, 15.0, 32.0, 45.0, 69.0, 70.0, 71.0, 72.0]
    data_sources = [sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times_1)),
                    sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times_2))]
    signal_source = sim.Population(1, sim.SpikeSourceArray(spike_times=signal_times))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    latches = MultipleNeuralLatchD(6, sim, global_params, neuron_params, std_conn, and_type="fast")

    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    latches.connect_constant_spikes(constant_spike_source.latch.output_neuron)
    latches.connect_data(data_sources, ini_pop_indexes=[0], component_indexes=[0, 1, 2])
    latches.connect_data(data_sources, ini_pop_indexes=[1], component_indexes=[3, 4, 5])

    latches.connect_signals(signal_source)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for latch in latches.latch_array:
        latch.latch_sr.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    out_spikes = []
    for latch in latches.latch_array:
        out_spikes.append(latch.latch_sr.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_latch_d", simtime)

    trace.printSpikes(1, "Data signal 1", data_up_times_1, "#FFC000")
    trace.printSpikes(2, "Data signal 2", data_up_times_2, "#FFC000")
    trace.printSpikes(3, "Store signal", signal_times, "#92D050")
    for i in range(len(latches.latch_array)):
        trace.printSpikes(4 + i, "FF" + str(i) + " spikes", out_spikes[i], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Latch): " + str(latches.total_neurons) +
          "\nNumber of total input connections (Latch): " + str(latches.total_input_connections) +
          "\nNumber of total internal connections (Latch): " + str(latches.total_internal_connections) +
          "\nNumber of total output connections (Latch): " + str(latches.total_output_connections))

    print(out_spikes)
