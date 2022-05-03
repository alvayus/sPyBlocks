import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.neural_oscillator import NeuralSyncOscillator
from sPyBlocks.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    n_clocks = 5
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    oscillators = [NeuralSyncOscillator(2**i, sim, global_params, neuron_params, std_conn) for i in range(n_clocks)]

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for i in range(n_clocks):
        oscillators[i].output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    spikes = []
    for i in range(n_clocks):
        spikes.append(oscillators[i].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_sync_oscillator", simtime)

    for i in range(n_clocks):
        trace.printSpikes(i + 1, "Output spikes (oscillator " + str(i) + ")", spikes[i][0], "#FFF2CC")

    trace.closeExcel()

    # Results
    total_neurons = 0
    total_input_connections = 0
    total_internal_connections = 0
    total_output_connections = 0

    for oscillator in oscillators:
        total_neurons += oscillator.total_neurons
        total_input_connections += oscillator.total_input_connections
        total_internal_connections += oscillator.total_internal_connections
        total_output_connections += oscillator.total_output_connections

    print("Number of total neurons (Sync. Oscillators): " + str(total_neurons) +
          "\nNumber of total input connections (Sync. Oscillators): " + str(total_input_connections) +
          "\nNumber of total internal connections (Sync. Oscillators): " + str(total_internal_connections) +
          "\nNumber of total output connections (Sync. Oscillators): " + str(total_output_connections))

    for i in range(n_clocks):
        print(spikes[i][0])

    times = range(0, int(simtime))
    len_times = len(times)

    values = np.zeros((n_clocks, len_times))

    for i in range(n_clocks):
        values[i, np.array(spikes[i]).astype(int)] = 1

    fig, axs = plt.subplots(5, 1, sharex='all', sharey='all')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.title('Clock signals')
    plt.xlabel('Time (ms)')
    plt.ylabel('Number of spikes')
    label = mpatches.Patch(color='red', label='Spike')
    plt.legend(handles=[label])

    for i in range(n_clocks):
        axs[i].plot(spikes[i], 1, 'ro', markersize=5)
        axs[i].step(times, values[i], where="post", color="darkmagenta")

    plt.show()
