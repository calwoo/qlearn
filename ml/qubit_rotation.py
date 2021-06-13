"""
Simple example of a parameterized quantum circuit, where here
we implement an example with two quantum rotation gates and train it
to output |1> from |0>.
"""

import pennylane as qml
from pennylane import numpy as np


# initialize qubit
dev1 = qml.device("default.qubit", wires=1)

# quantum circuit
@qml.qnode(dev1)
def circuit(params):
    phi1, phi2 = params
    # perform a parameterized rotation around bloch x-axis
    qml.RX(phi1, wires=0)
    # rotate around bloch y-axis
    qml.RY(phi2, wires=0)
    # compute expectation value of pauli z-operator
    return qml.expval(qml.PauliZ(0))

# evaluate circuit
print(circuit([0.54, 0.12]))
