import spynnaker8 as sim

from sPyBlocks import MultipleNeuralFlipFlop
from sPyBlocks import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 50.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"tau_m": 0.1, "v_rest": -65.0, "v_thresh": -64.94, "tau_syn_E": 1.0, "tau_syn_I": 1.0,
                     "tau_refrac": 0.0}

    # Network building
    spike_source_1 = sim.Population(1, sim.SpikeSourceArray(spike_times=[1.0]))
    spike_source_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=[26.0]))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    flip_flops = MultipleNeuralFlipFlop(3, sim, global_params, neuron_params, std_conn)

    # Testing
    flip_flops.connect_set(spike_source_1, gate_indexes=[0])
    flip_flops.connect_set(spike_source_2, gate_indexes=[2])

    for flip_flop in flip_flops.latch_array:
        flip_flop.input_neuron.record(('spikes'))
        flip_flop.cycle_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    input_spikes = []
    cycle_spikes = []
    for flip_flop in flip_flops.latch_array:
        input_spikes.append(flip_flop.input_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        cycle_spikes.append(flip_flop.cycle_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("trace_functions_test", simtime)
    for i in range(len(flip_flops.latch_array)):
        trace.printSpikes(i * 2 + 1, "FF" + str(i) + " (Input)", input_spikes[i][0], "#FFF2CC")
        trace.printSpikes(i * 2 + 2, "FF" + str(i) + " (Cycle)", cycle_spikes[i][0], "#FFF2CC")

    trace.closeExcel()
