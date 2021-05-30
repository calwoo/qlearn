"""
Circuit for creating a maximally-entangled Bell state
"""

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram


# creates the bell state 1/sqrt(2) (|00> + |11>)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

qc.draw(output="mpl")
plt.show()

# simulate a bell state
sim = Aer.get_backend("statevector_simulator")
qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()

plot_histogram(counts)
plt.show()

# this circuit instead creates the bell state 1/sqrt(2) (|01> + |10>)
qc_alt = QuantumCircuit(2)
qc_alt.h(0)
qc_alt.cx(0, 1)
qc_alt.x(1)

qc_alt.draw(output="mpl")
plt.show()

qobj = assemble(qc_alt)
counts = sim.run(qobj).result().get_counts()

plot_histogram(counts)
plt.show()
