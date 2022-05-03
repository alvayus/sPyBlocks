import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import spynnaker8 as sim

from sPyBlocks.neural_or import MultipleNeuralOr
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 20.0  # (ms)

    # Other parameters
    show_graph = False
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times = range(1, 16)
    spike_sources = sim.Population(2, sim.SpikeSourceArray(spike_times=spike_times))

    spike_sources_array = []
    for i in range(2):
        spike_sources_array.append(sim.PopulationView(spike_sources, [i]))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    or_gates = MultipleNeuralOr(6, sim, global_params, neuron_params, std_conn)

    # Testing
    or_gates.connect_inputs(spike_sources, ini_pop_indexes=[[], [0], [0, 1]], component_indexes=[0, 1, 2])
    or_gates.connect_inputs(spike_sources_array, ini_pop_indexes=[[], [0], [0, 1]], component_indexes=[3, 4, 5])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for gate in or_gates.or_array:
        gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    spikes = []
    voltage = []
    for gate in or_gates.or_array:
        spikes.append(gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        voltage.append(gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_or", simtime)

    trace.printSpikes(1, "Spike source 0", spike_times, "#92D050")
    trace.printSpikes(2, "Spike source 1", spike_times, "#92D050")

    for i in range(len(or_gates.or_array)):
        trace.printSpikes(3 + i, "OR response (gate " + str(i) + ")", spikes[i][0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (OR gates): " + str(or_gates.total_neurons) +
          "\nNumber of total input connections (OR gates): " + str(or_gates.total_input_connections) +
          "\nNumber of total internal connections (OR gates): " + str(or_gates.total_internal_connections) +
          "\nNumber of total output connections (OR gates): " + str(or_gates.total_output_connections))

    if show_graph:
        for i in range(len(or_gates.or_array)):
            print(spikes[i])

            plt.plot(voltage[i].times, voltage[i])
            plt.plot(spikes[i], [neuron_params["v_rest"]] * len(spikes[i]), 'ro', markersize=5)
            plt.vlines(spikes[i], neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
            plt.title('OR neuron response (gate ' + str(i) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            or_label = mpatches.Patch(color='red', label='OR neuron spike')
            plt.legend(handles=[or_label])
            plt.show()
