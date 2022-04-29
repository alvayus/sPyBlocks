import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from neural_logic_gates.neural_oscillator import NeuralSyncOscillator
from neural_logic_gates.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 20.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    oscillator = NeuralSyncOscillator(2, sim, global_params, neuron_params, std_conn)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    oscillator.input_neuron.record(('spikes', 'v'))
    oscillator.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    in_spikes = oscillator.input_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    in_voltage = oscillator.input_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]
    out_spikes = oscillator.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    out_voltage = oscillator.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_sync_oscillator", simtime)

    trace.printSpikes(1, "Input spikes", in_spikes[0], "#FFF2CC")
    trace.printSpikes(2, "Output spikes", out_spikes[0], "#FFF2CC")

    values = np.zeros(int(simtime))
    for i in range(len(values)):
        if i in out_spikes[0]:
            values[i] = 1
    trace.printRow(4, "OUTPUT", values, "#FFC000")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Sync. Oscillator): " + str(oscillator.total_neurons) +
          "\nNumber of total input connections (Sync. Oscillator): " + str(oscillator.total_input_connections) +
          "\nNumber of total internal connections (Sync. Oscillator): " + str(oscillator.total_internal_connections) +
          "\nNumber of total output connections (Sync. Oscillator): " + str(oscillator.total_output_connections))

    print(in_spikes)
    print(out_spikes)

    plt.subplot(121)
    plt.plot(in_voltage.times, in_voltage)
    plt.plot(in_spikes, [neuron_params["v_rest"]] * len(in_spikes), 'ro', markersize=5)
    plt.vlines(in_spikes, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('Input neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    in_label = mpatches.Patch(color='red', label='Input neuron spike')
    plt.legend(handles=[in_label])

    times = range(0, int(simtime))
    len_times = len(times)
    out_values = np.zeros(len_times)
    out_values[np.array(out_spikes).astype(int)] = 1

    plt.subplot(122)
    #plt.plot(out_voltage.times, out_voltage)
    plt.plot(times, out_values, 'ro', markersize=5)
    plt.step(times, out_values, where="post", color="darkmagenta")
    plt.title('Output neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    out_label = mpatches.Patch(color='red', label='Output neuron spike')
    plt.legend(handles=[out_label])

    plt.show()
