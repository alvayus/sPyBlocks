import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spynnaker8 as sim

from neural_logic_gates.constant_spike_source import ConstantSpikeSource
from neural_logic_gates.neural_not import NeuralNot
from neural_logic_gates.trace_functions import SpikeTrace

if __name__ == "__main__":
    # Simulator initialization and simulation params
    sim.setup(timestep=1.0)
    simtime = 100.0  # (ms)

    # Other parameters
    global_params = {"min_delay": 1.0}
    neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1,
                     "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}

    # Network building
    spike_times = [15.0, 17.0, 19.0, 21.0]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])

    not_gate = NeuralNot(sim, global_params, neuron_params, std_conn)
    constant_spike_source = ConstantSpikeSource(sim, global_params, neuron_params, std_conn)

    # Testing
    not_gate.connect_inputs(spike_source)
    not_gate.connect_excitation([constant_spike_source.set_source, constant_spike_source.latch.output_neuron])

    # --- NO NEED TO TOUCH THIS PART OF THE CODE ---
    constant_spike_source.latch.output_neuron.record(('spikes', 'v'))
    not_gate.output_neuron.record(('spikes', 'v'))

    # Run simulation
    sim.run(simtime)

    # Data from the simulation
    css_set_spike = [1]
    css_ff_spikes = constant_spike_source.latch.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains

    out_spikes = not_gate.output_neuron.get_data(variables=["spikes"]).segments[0].spiketrains
    out_voltage = not_gate.output_neuron.get_data(variables=["v"]).segments[0].analogsignals[0]

    # End simulation
    sim.end()

    # Excel file creation
    trace = SpikeTrace("test_simple_not", simtime)

    trace.printSpikes(1, "Input signal", spike_times, "#92D050")
    trace.printSpikes(2, "Constant spike source (Set)", css_set_spike, "#FF0000")
    trace.printSpikes(3, "Constant spike source (FF)", css_ff_spikes[0], "#FF0000")
    trace.printSpikes(4, "NOT response", out_spikes[0], "#FFF2CC")

    trace.closeExcel()

    # Results
    print("Number of total neurons (NOT gate): " + str(not_gate.total_neurons) +
          "\nNumber of total input connections (NOT gate): " + str(not_gate.total_input_connections) +
          "\nNumber of total internal connections (NOT gate): " + str(not_gate.total_internal_connections) +
          "\nNumber of total output connections (NOT gate): " + str(not_gate.total_output_connections))

    print(out_spikes)

    plt.plot(out_voltage.times, out_voltage)
    plt.plot(out_spikes, [neuron_params["v_rest"]] * len(out_spikes), 'ro', markersize=5)
    plt.vlines(out_spikes, neuron_params["v_rest"], neuron_params["v_thresh"], colors="red")
    plt.title('Output neuron response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    out_label = mpatches.Patch(color='red', label='Output neuron spike')
    plt.legend(handles=[out_label])

    plt.show()
