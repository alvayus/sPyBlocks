import numpy as np
import spynnaker8 as sim

from neural_logic_gates.neural_oscillator import NeuralAsyncOscillator
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
    oscillator = NeuralAsyncOscillator(sim, global_params, neuron_params, std_conn)

    # Testing
    spike_times = [1.0, 6.0, 7.0, 8.0, 10.0, 12.0, 20.0]
    signal_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    oscillator.connect_signal(signal_source)

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    oscillator.input_neuron.record(('spikes'))
    oscillator.cycle_neuron.record(('spikes'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    in_spikes = oscillator.input_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    cycle_spikes = oscillator.cycle_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    total_spikes = np.concatenate([in_spikes[0], cycle_spikes[0]])

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_async_oscillator", simtime)

    trace.printSpikes(1, "Change signal", spike_times, "#FFC000")
    trace.printSpikes(2, "Input neuron", in_spikes[0], "#FFF2CC")
    trace.printSpikes(3, "Cycle neuron", cycle_spikes[0], "#FFF2CC")
    trace.printSpikes(6, "OUTPUT", total_spikes, "#FFC000")

    trace.closeExcel()

    # Results
    print("Number of total neurons (Async. Oscillator): " + str(oscillator.total_neurons) +
          "\nNumber of total input connections (Async. Oscillator): " + str(oscillator.total_input_connections) +
          "\nNumber of total internal connections (Async. Oscillator): " + str(oscillator.total_internal_connections) +
          "\nNumber of total output connections (Async. Oscillator): " + str(oscillator.total_output_connections))

    print(in_spikes)
    print(cycle_spikes)
