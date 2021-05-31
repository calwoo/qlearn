"""
Construction of 4-qubit W-state.
"""

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram


qc = QuantumCircuit(4)
qc.h(0)
qc.h(3)
qc.x(0)
qc.x(3)
qc.ccx(0, 3, 1)
qc.x(0)
qc.x(3)
qc.ccx(0, 3, 2)
qc.cx(2, 0)
qc.cx(2, 3)
qc.draw(output="mpl")
plt.show()

# simulate
sim = Aer.get_backend("statevector_simulator")
qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()

plot_histogram(counts)
plt.show()