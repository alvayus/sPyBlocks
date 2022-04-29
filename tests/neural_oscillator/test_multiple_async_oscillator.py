import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from neural_logic_gates.connection_functions import create_connections
from neural_logic_gates.neural_oscillator import MultipleNeuralAsyncOscillator
from neural_logic_gates.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization neural_and simulation params
    sim.setup(timestep=1.0)
    simtime = 10.0  # (ms)

    # Other parameters
    n_oscillators = 4
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    set_source = sim.Population(1, sim.SpikeSourceArray(spike_times=[1.0]))

    spike_times_1 = np.arange(1.0, simtime, global_params["min_delay"])
    spike_source_1 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_1))
    spike_times_2 = np.arange(1.0, simtime, global_params["min_delay"] * 2)
    spike_source_2 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_2))
    spike_times_3 = np.arange(simtime / 2, simtime, global_params["min_delay"])
    spike_source_3 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_3))
    spike_times_4 = np.arange(simtime / 2, simtime, global_params["min_delay"] * 2)
    spike_source_4 = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_4))

    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])
    switches = MultipleNeuralAsyncOscillator(n_oscillators, sim, global_params, neuron_params, std_conn)

    # Testing
    switches.connect_signal(spike_source_1, component_indexes=[0])
    switches.connect_signal(spike_source_2, component_indexes=[1])
    switches.connect_signal(spike_source_3, component_indexes=[2])
    switches.connect_signal(spike_source_4, component_indexes=[3])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    for oscillator in switches.oscillator_array:
        oscillator.input_neuron.record(('spikes'))
        oscillator.cycle_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    in_spikes = []
    cycle_spikes = []
    total_spikes = []
    for i in range(n_oscillators):
        in_spikes.append(switches.oscillator_array[i].input_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        cycle_spikes.append(switches.oscillator_array[i].cycle_neuron.get_data(variables=["spikes"]).segments[0].spiketrains)
        total_spikes.append(np.concatenate([in_spikes[i][0], cycle_spikes[i][0]]))

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_multiple_async_oscillator", simtime)

    trace.printSpikes(1, "Input signal 1", spike_times_1, "#92D050")
    trace.printSpikes(2, "Input signal 2", spike_times_2, "#92D050")
    trace.printSpikes(3, "Input signal 3", spike_times_3, "#92D050")
    trace.printSpikes(4, "Input signal 4", spike_times_4, "#92D050")

    for i in range(n_oscillators):
        trace.printSpikes(5 + i * 3, 'Input neuron response (oscillator ' + str(i) + ')', in_spikes[i][0], "#FFF2CC")
        trace.printSpikes(5 + i * 3 + 1, 'Cycle neuron response (oscillator ' + str(i) + ')', cycle_spikes[i][0], "#FFF2CC")
        trace.printSpikes(5 + i * 3 + 2, 'Total response (oscillator ' + str(i) + ')', total_spikes[i], "#FFC000")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Async. Oscillators): " + str(switches.total_neurons) +
          "\nNumber of total input connections (Async. Oscillators): " + str(switches.total_input_connections) +
          "\nNumber of total internal connections (Async. Oscillators): " + str(switches.total_internal_connections) +
          "\nNumber of total output connections (Async. Oscillators): " + str(switches.total_output_connections))

    for i in range(n_oscillators):
        print(in_spikes[i])
        print(cycle_spikes[i])