import matplotlib.pyplot as plt
from math import pi, sqrt

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram


def x_measurement(qc, qubit, cbit):
    """
    Perform a measurement in the X-basis |+>, |->
    and store result in cbit
    """

    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc


initial_state = [1/sqrt(2), -1/sqrt(2)]
qc = QuantumCircuit(1, 1)
qc.initialize(initial_state, 0)

# |-> to |1> via hadamard
qc.h(0)


# measure in x-basis
x_measurement(qc, 0, 0)
qc.draw(output="mpl")
plt.show()

# we expect |1> = 1/sqrt(2) |+> - 1/sqrt(2) |->

# simulate
sim = Aer.get_backend("qasm_simulator")
qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)
plt.show()
