import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import spynnaker8 as sim

from sPyBlocks.neural_or import NeuralOr

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 10.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=[1.0]))
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    or_gate = NeuralOr(sim, global_params, neuron_params, std_conn)

    # Testing
    test = [1, 0, 0, 0, 0]
    or_conn = sim.StaticSynapse(weight=sum(test), delay=global_params["min_delay"])
    or_gate.connect_inputs(spike_source, or_conn)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    or_gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    or_spikes = or_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    or_voltage = or_gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Results
    print("Number of total neurons (OR gate): " + str(or_gate.total_neurons) +
          "\nNumber of total input connections (OR gate): " + str(or_gate.total_input_connections) +
          "\nNumber of total internal connections (OR gate): " + str(or_gate.total_internal_connections) +
          "\nNumber of total output connections (OR gate): " + str(or_gate.total_output_connections))

    print(or_spikes)

    plt.plot(or_voltage.times, or_voltage)
    plt.plot(or_spikes, [neuron_params["v_rest"]] * len(or_spikes), 'ro', markersize=5)
    plt.vlines(or_spikes, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('OR neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    or_label = mpatches.Patch(color='red', label='OR neuron spike')
    plt.legend(handles=[or_label])
    plt.show()
