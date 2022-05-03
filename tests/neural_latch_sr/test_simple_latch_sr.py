import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import spynnaker8 as sim

from sPyBlocks.neural_latch_sr import NeuralLatchSR
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
    set_times = [1.0, 21.0]
    reset_times = [11.0, 81.0]
    set_source = sim.Population(1, sim.SpikeSourceArray(spike_times=set_times))
    reset_source = sim.Population(1, sim.SpikeSourceArray(spike_times=reset_times))
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    flip_flop = NeuralLatchSR(sim, global_params, neuron_params, std_conn)

    # Testing
    test = 1
    if test:
        flip_flop.connect_set(set_source)
        flip_flop.connect_reset(reset_source)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    flip_flop.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    out_spikes = flip_flop.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    out_voltage = flip_flop.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_flipflop_sr", simtime)

    trace.printSpikes(1, "Set signal", set_times, "#92D050")
    trace.printSpikes(2, "Reset times", reset_times, "#FF0000")
    trace.printSpikes(3, "FF output neuron", out_spikes[0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Flip-flop): " + str(flip_flop.total_neurons) +
          "\nNumber of total input connections (Flip-flop): " + str(flip_flop.total_input_connections) +
          "\nNumber of total internal connections (Flip-flop): " + str(flip_flop.total_internal_connections) +
          "\nNumber of total output connections (Flip-flop): " + str(flip_flop.total_output_connections))

    print(out_spikes)

    plt.plot(out_voltage.times, out_voltage)
    plt.plot(out_spikes, [neuron_params["v_rest"]] * len(out_spikes), 'ro', markersize=5)
    plt.vlines(out_spikes, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('Input neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    in_label = mpatches.Patch(color='red', label='Input neuron spike')
    plt.legend(handles=[in_label])

    plt.show()
