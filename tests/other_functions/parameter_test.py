import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import spynnaker8 as sim

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 2.0}

    old_params = {"cm": 0.265, "tau_m": 10.0, "tau_refrac": 1.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                  "v_rest": -70.0, "v_reset": -70.0, "v_thresh": -69.0}

    new_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 1.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                  "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    non_ideal_params = {"tau_m": 0.1, "v_rest": -65.0, "v_thresh": -64.94, "tau_syn_E": 1.0, "tau_syn_I": 1.0}
    non_ideal_params_2 = {"tau_m": 0.1, "v_rest": -65.0, "v_thresh": -64.94}

    # Network building
    params = non_ideal_params_2
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=[1.0, 49.0]))
    reset_source = sim.Population(1, sim.SpikeSourceArray(spike_times=[49.0, 51.0]))
    test_neuron = sim.Population(1, sim.IF_curr_exp(**params), initial_values={'v': params["v_rest"]})

    # Testing
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    sim.Projection(spike_source, test_neuron, sim.AllToAllConnector(), std_conn, receptor_type="excitatory")
    sim.Projection(reset_source, test_neuron, sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    test_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    spikes = test_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    voltage = test_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Results
    print(spikes)
    print([min(voltage), max(voltage)])

    plt.plot(voltage.times, voltage)
    plt.plot(spikes, [params["v_rest"]] * len(spikes), 'ro', markersize=5)
    plt.title('Test neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    label = mpatches.Patch(color='red', label='Test neuron spike')
    plt.legend(handles=[label])

    plt.show()

    # Notes:
    # - All parameters are working when you comply with two restrictions:
    #   * You have to make sure your neurons fire 1 time when you send 1 input spike
    #   * You have to make sure that you are sending input spikes to neurons that are at resting potential
    #     (therefore you have to wait for the neuron to reach resting potential before sending a new spike)
