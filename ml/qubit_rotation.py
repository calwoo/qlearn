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

# gradient
dcircuit = qml.grad(circuit, argnum=0)

def cost(x):
    return circuit(x)

# optimize
params = np.array([0.011, 0.012])
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100

for i in range(steps):
    params = opt.step(cost, params)
    if (i+1) % 5 == 0:
        print(f"epoch {i+1} cost: {cost(params)}")

print(f"trained params: {params}")
