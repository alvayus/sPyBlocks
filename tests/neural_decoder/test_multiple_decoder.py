# TODO: Test not implemented as the multiple decoder object
'''
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from sPyBlocks.neural_decoder import MultipleNeuralDecoder

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 20.0  # (ms)

    # Other parameters
    n_components = 3
    n_inputs = 2
    # It may be possible that you need to increase the time_scale_factor to receive all the packets when you're working
    # with a number of inputs greater than 4
    global_params = {"min_delay": 2.0}
    neuron_params = {"cm": 0.265, "tau_m": 10.0, "tau_refrac": 1.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                     "v_rest": -70.0, "v_reset": -70.0, "v_thresh": -69.0}

    # Network building
    spike_times = np.arange(1.0, simtime, global_params["min_delay"])
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    decoders = MultipleNeuralDecoder(n_components, n_inputs, sim, global_params, neuron_params, std_conn)

    # Testing
    tests = [[0, 0], [0, 1], [1, 1]]
    for i in range(len(tests)):
        flipped_test = np.flip(tests[i])
        connect_indexes = np.nonzero(flipped_test)[0]
        decoders.decoder_array[i].connect_inputs(spike_source, gate_indexes=connect_indexes)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for decoder in decoders.decoder_array:
        for gate in decoder.and_gates.and_array:
            gate.and_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    out_spikes = []
    for i in range(n_components):
        decoder_response = []
        for j in range(2 ** n_inputs):
            decoder_response.append(decoders.decoder_array[i].and_gates.and_array[j].and_neuron
                                    .get_data(variables=["spikes"]).segments[0].spiketrains[0])
        out_spikes.append(decoder_response)

    # End simulation
    sim.end()

    # Results
    out_lengths = np.zeros(2 ** n_inputs, dtype=int)
    for decoder_response in out_spikes:
        for i in range(2 ** n_inputs):
            if len(decoder_response[i]) != 0:
                out_lengths[i] += 1

    print(out_spikes)

    fig = plt.figure()
    for i in range(n_components):
        ax = fig.add_subplot(1, n_components, i+1)
        plt.bar(range(len(tests[i])), tests[i], color="tab:blue")
        plt.title('Input tests (decoder ' + str(i) + ")")
        plt.xlabel('Test array indexes')
        plt.ylabel('Value')
        plt.xticks(range(len(tests[i])))
        plt.yticks([0, 1])

    plt.figure()
    plt.bar(range(len(out_lengths)), out_lengths, color="tab:red")
    plt.title('AND neuron responses')
    plt.xlabel('AND neuron indexes')
    plt.ylabel('Number of decoders firing in that AND neuron')
    plt.xticks(range(len(out_lengths)))
    plt.yticks(range(max(out_lengths)+1))

    plt.show()
    '''