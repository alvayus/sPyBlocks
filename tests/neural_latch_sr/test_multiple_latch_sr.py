import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import spynnaker8 as sim

from sPyBlocks.neural_latch_sr import MultipleNeuralLatchSR
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    show_graph = False
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    set_times = [1.0]
    reset_times = [61.0]
    spike_sources = sim.Population(2, sim.SpikeSourceArray(spike_times=set_times))
    spike_sources_array = [sim.PopulationView(spike_sources, [0])]
    reset_source = sim.Population(1, sim.SpikeSourceArray(spike_times=reset_times))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    flip_flops = MultipleNeuralLatchSR(6, sim, global_params, neuron_params, std_conn)

    # Testing
    flip_flops.connect_set(spike_sources, ini_pop_indexes=[[0], [1]], component_indexes=[0, 3])
    flip_flops.connect_set(spike_sources_array, component_indexes=[1])
    flip_flops.connect_reset(reset_source, component_indexes=[3])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for flip_flop in flip_flops.latch_array:
        flip_flop.output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    spikes = []
    for flip_flop in flip_flops.latch_array:
        spikes.append(flip_flop.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_flipflop_sr", simtime)

    trace.printSpikes(1, "Set signal", set_times, "#92D050")
    trace.printSpikes(2, "Reset times", reset_times, "#FF0000")
    for i in range(len(flip_flops.latch_array)):
        trace.printSpikes(3 + i, "FF" + str(i) + " spikes", spikes[i][0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Flip-flops): " + str(flip_flops.total_neurons) +
          "\nNumber of total input connections (Flip-flops): " + str(flip_flops.total_input_connections) +
          "\nNumber of total internal connections (Flip-flops): " + str(flip_flops.total_internal_connections) +
          "\nNumber of total output connections (Flip-flops): " + str(flip_flops.total_output_connections))

    if show_graph:
        for i in range(len(flip_flops.latch_array)):
            print(spikes[i])

            plt.plot(spikes[i], [neuron_params["v_rest"]] * len(spikes[i]), 'ro', markersize=5)
            plt.vlines(spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
            plt.title('OR neuron response (gate ' + str(i) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            or_label = mpatches.Patch(color='red', label='OR neuron spike')
            plt.legend(handles=[or_label])
            plt.show()