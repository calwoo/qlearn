### Phase kickback

There are two heuristic descriptions of phase kickback out there, and I want to collect/compare them.

First one from the IBM Qiskit book is: "Kickback is where the eigenvalue added by a gate to a qubit is ‘kicked back’ into a different qubit via a controlled operation."

To explain, the eigenvalue added to a qubit has norm 1, and is known as a **phase**. For example, the $X$-gate on the $|->$ qubit gives

$$ X|-> = -|-> $$

which is a phase of -1. This is a global phase, and so has no observable effects.

However, let's investigate this behavior on a controlled-NOT gate. We compute that

$$ CNOT|-0> = |-0> $$

and 

$$ CNOT|-1> = -|-1> $$

which is also a global phase. However in superposition we see that

$$ CNOT|-+> = |--> $$

Considering that CNOT normally takes $CNOT|01> = |11>$, i.e. it leaves the control qubit undisturbed, this is interesting! Here, the control qubit is the one that changes, not the target qubit. That is, wrapping the computational basis qubits in Hadamard gates allows one to "flip" the CNOT gate.

This is useful, as sometimes in quantum hardware we can only build CNOT gates in a single direction. This trick allows us to get around this difficulty.