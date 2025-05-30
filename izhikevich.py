import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.utils.ensemble import tuning_curves
from nengo.utils.matplotlib import rasterplot

with nengo.Network(seed=0) as model:
    u = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    ens = nengo.Ensemble(10, dimensions=1, neuron_type=nengo.Izhikevich())
    nengo.Connection(u, ens)


with model:
    out_p = nengo.Probe(ens, synapse=0.03)
    spikes_p = nengo.Probe(ens.neurons)
    voltage_p = nengo.Probe(ens.neurons, "voltage")
    recovery_p = nengo.Probe(ens.neurons, "recovery")


with nengo.Simulator(model) as sim:
    sim.run(1.0)

t = sim.trange()
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, sim.data[out_p])
plt.ylabel("Decoded output")
ax = plt.subplot(2, 1, 2)
rasterplot(t, sim.data[spikes_p], ax=ax)
plt.ylabel("Neuron")

def izh_plot(sim):
    t = sim.trange()
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(t, sim.data[out_p])
    plt.ylabel("Decoded output")
    plt.xlim(right=t[-1])
    ax = plt.subplot(4, 1, 2)
    rasterplot(t, sim.data[spikes_p], ax=ax)
    plt.ylabel("Neuron")
    plt.xlim(right=t[-1])
    plt.subplot(4, 1, 3)
    plt.plot(t, sim.data[voltage_p])
    plt.ylabel("Voltage")
    plt.xlim(right=t[-1])
    plt.subplot(4, 1, 4)
    plt.plot(t, sim.data[recovery_p])
    plt.ylabel("Recovery")
    plt.xlim(right=t[-1])


u.output = 0.2
with nengo.Simulator(model) as sim:
    sim.run(0.1)
izh_plot(sim)

with nengo.Network(seed=0) as model:
    u = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    ens = nengo.Ensemble(30, dimensions=1, neuron_type=nengo.Izhikevich())
    nengo.Connection(u, ens)
    out_p = nengo.Probe(ens, synapse=0.03)


with nengo.Simulator(model) as sim:
    plt.figure()
    plt.plot(*tuning_curves(ens, sim))