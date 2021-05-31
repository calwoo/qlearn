"""
Construction of n-qubit GHZ-state.
"""

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram


n = 8
qc = QuantumCircuit(n)
# initialize superposition in first qubit
qc.h(0)
# set all other qubit states using CNOT
for i in range(n-1):
    qc.cx(i, i+1)

qc.draw(output="mpl")
plt.show()

# simulate
sim = Aer.get_backend("statevector_simulator")
qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()

plot_histogram(counts)
plt.show()