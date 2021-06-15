import pennylane as qml
import numpy as np

import torch
import torch.nn as nn


n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

def real(angles, **kwargs):
    qml.Hadamard(wires=0)
    qml.Rot(*angles, wires=0)

def generator(w, **kwargs):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)

def discriminator(w):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=2)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=2)
    qml.CNOT(wires=[0, 2])
    qml.RX(w[6], wires=2)
    qml.RY(w[7], wires=2)
    qml.RZ(w[8], wires=2)

# qnodes
@qml.qnode(dev, interface="torch")
def real_disc_circuit(phi, theta, omega, disc_w):
    real([phi, theta, omega])
    discriminator(disc_w)
    return qml.expval(qml.PauliZ(2))

@qml.qnode(dev, interface="torch")
def gen_disc_circuit(gen_w, disc_w):
    generator(gen_w)
    discriminator(disc_w)
    return qml.expval(qml.PauliZ(2))

# cost functions
def prob_real_true(disc_w):
    true_disc_output = real_disc_circuit(phi, theta, omega, disc_w)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true

def prob_fake_true(gen_w, disc_w):
    fake_disc_output = gen_disc_circuit(gen_w, disc_w)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true

def disc_cost(disc_w):
    cost = prob_fake_true(gen_w, disc_w) - prob_real_true(disc_w)
    return cost

def gen_cost(gen_w, disc_w):
    return -prob_fake_true(gen_w, disc_w)

phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
np.random.seed(0)
eps = 1e-2
init_gen_w = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=(9,))
init_disc_w = np.random.normal(size=(9,))

gen_w = torch.tensor(init_gen_w, requires_grad=True)
disc_w = torch.tensor(init_disc_w, requires_grad=True)

# train discriminator
opt_d = torch.optim.SGD([disc_w], lr=0.4)
for step in range(50):
    loss = disc_cost(disc_w)

    opt_d.zero_grad()
    loss.backward()
    opt_d.step()

    if step % 5 == 0:
        loss_val = loss.item()
        print("Step {}: cost = {}".format(step, loss_val))

print("Prob(real classified as real): ", prob_real_true(disc_w).detach().numpy())
print("Prob(fake classified as real): ", prob_fake_true(gen_w, disc_w).detach().numpy())

# train generator
opt_g = torch.optim.SGD([gen_w], lr=0.4)
for step in range(50):
    loss = gen_cost(gen_w, disc_w)

    opt_g.zero_grad()
    loss.backward()
    opt_g.step()

    if step % 5 == 0:
        loss_val = loss.item()
        print("Step {}: cost = {}".format(step, loss_val))

print("Prob(fake classified as real): ", prob_fake_true(gen_w, disc_w).detach().numpy())
print("Discriminator cost: ", disc_cost(disc_w).detach().numpy())

# testing
obs = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]

bloch_vector_real = qml.map(real, obs, dev, interface="torch")
bloch_vector_generator = qml.map(generator, obs, dev, interface="torch")

print("Real Bloch vector: {}".format(bloch_vector_real([phi, theta, omega])))
print("Generator Bloch vector: {}".format(bloch_vector_generator(gen_w)))
