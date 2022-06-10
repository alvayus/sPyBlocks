import spynnaker8 as sim

from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_latch_d import NeuralLatchD
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
    data_up_times = [10.0, 20.0, 45.0, 70.0, 72.0]
    signal_times = [10.0, 15.0, 32.0, 45.0, 69.0, 70.0, 71.0, 72.0]
    data_source = sim.Population(1, sim.SpikeSourceArray(spike_times=data_up_times))
    signal_source = sim.Population(1, sim.SpikeSourceArray(spike_times=signal_times))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    latch = NeuralLatchD(sim, global_params, neuron_params, std_conn, and_type="fast")

    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    latch.connect_constant_spikes(constant_spike_source.latch.output_neuron)
    latch.connect_data(data_source)
    latch.connect_signal(signal_source)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    latch.not_gate.output_neuron.record(('spikes'))

    for gate in latch.and_gates.and_array:
        if gate.build_type == "classic":
            gate.or_gate.output_neuron.record(('spikes'))
        gate.output_neuron.record(('spikes'))

    latch.latch_sr.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    not_spikes = latch.not_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0]
    and_spikes = [gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0] for gate in
                  latch.and_gates.and_array]
    out_spikes = latch.latch_sr.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains[0]

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_latch_d", simtime)

    trace.printSpikes(1, "Data signal", data_up_times, "#FFC000")
    trace.printSpikes(2, "Store signal", signal_times, "#92D050")
    trace.printSpikes(3, "NOT response", not_spikes, "#FF0000")
    trace.printSpikes(4, "AND response (S)", and_spikes[0], "#FFC000")
    trace.printSpikes(5, "AND response (RS)", and_spikes[1], "#FFC000")
    trace.printSpikes(6, "Latch output neuron", out_spikes, "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Latch): " + str(latch.total_neurons) +
          "\nNumber of total input connections (Latch): " + str(latch.total_input_connections) +
          "\nNumber of total internal connections (Latch): " + str(latch.total_internal_connections) +
          "\nNumber of total output connections (Latch): " + str(latch.total_output_connections))

    print(out_spikes)
