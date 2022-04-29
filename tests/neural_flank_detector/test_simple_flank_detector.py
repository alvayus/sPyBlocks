import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from neural_logic_gates.constant_spike_source import ConstantSpikeSource
from neural_logic_gates.neural_flank_detector import NeuralFlankDetector
from neural_logic_gates.neural_oscillator import NeuralSyncOscillator
from neural_logic_gates.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 30.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    oscillator = NeuralSyncOscillator(3, sim, global_params, neuron_params, std_conn)
    flank_detector = NeuralFlankDetector(sim, global_params, neuron_params, std_conn)
    constant_signal_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    flank_detector.connect_inputs(oscillator.output_neuron)
    flank_detector.connect_constant_spikes([constant_signal_source.set_source, constant_signal_source.latch.output_neuron])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    oscillator.output_neuron.record(('spikes'))
    flank_detector.not_gate.output_neuron.record(('spikes'))
    flank_detector.and_gates.and_array[0].output_neuron.record(('spikes'))
    flank_detector.and_gates.and_array[1].output_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    clock_spikes = oscillator.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    not_spikes = flank_detector.not_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    rising_edges = flank_detector.and_gates.and_array[0].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    falling_edges = flank_detector.and_gates.and_array[1].output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_flank_detector", simtime)

    trace.printSpikes(1, "Clock signal", clock_spikes[0], "#92D050")
    trace.printSpikes(2, "NOT response", not_spikes[0], "#FF0000")
    trace.printSpikes(3, "Rising edges", rising_edges[0], "#FFF2CC")
    trace.printSpikes(4, "Falling edges", falling_edges[0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Flank detector): " + str(flank_detector.total_neurons) +
          "\nNumber of total input connections (Flank detector): " + str(flank_detector.total_input_connections) +
          "\nNumber of total internal connections (Flank detector): " + str(flank_detector.total_internal_connections) +
          "\nNumber of total output connections (Flank detector): " + str(flank_detector.total_output_connections))

    print(clock_spikes)
    print(not_spikes)
    print(rising_edges)
    print(falling_edges)

    times = range(0, int(simtime))
    len_times = len(times)

    clock_values = np.zeros(len_times)
    clock_values[np.array(clock_spikes).astype(int)] = 1

    fig, axs = plt.subplots(4, 1, sharex='all', sharey='all')
    fig.suptitle('Flank detector test')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Time (ms)')
    plt.ylabel('Number of spikes')
    spike_label = mpatches.Patch(color='red', label='Spike')
    rising_label = mpatches.Patch(color='goldenrod', label='Rising edge')
    falling_label = mpatches.Patch(color='teal', label='Falling edge')
    plt.legend(handles=[spike_label, rising_label, falling_label])

    axs[0].set_title("Original clock spikes")
    axs[0].plot(clock_spikes, 1, color="red", markersize=5, marker='o')
    axs[0].step(times, clock_values, where="post", color="darkmagenta")

    axs[1].set_title("Inverted clock spikes")
    axs[1].step(np.array(times) + std_conn.delay + flank_detector.not_gate.delay, clock_values, where="post", color="darkmagenta")
    axs[1].plot(not_spikes, 1, 'ro', markersize=5)
    axs[1].vlines(not_spikes, 0, 1, colors="red")

    axs[2].set_title("Rising edges (output of AND neuron 0)")
    axs[2].step(times, clock_values, where="post", color="thistle")
    axs[2].step(np.array(times) + flank_detector.rising_delay, clock_values, where="post", color="darkmagenta")
    axs[2].plot(rising_edges, 1, color="goldenrod", markersize=5, marker='o')
    axs[2].vlines(rising_edges, 0, 1, colors="goldenrod")

    axs[3].set_title("Falling edges (output of AND neuron 1)")
    axs[3].step(times, clock_values, where="post", color="thistle")
    axs[3].step(np.array(times) + flank_detector.falling_delay, clock_values, where="post", color="darkmagenta")
    axs[3].plot(falling_edges, 1, color="teal", markersize=5, marker='o')
    axs[3].vlines(falling_edges, 0, 1, colors="teal")

    plt.show()
