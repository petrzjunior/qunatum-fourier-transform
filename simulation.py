import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

circuit = QuantumCircuit(4)
circuit.initialize(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]) / np.sqrt(4), circuit.qubits)
circuit.append(QFT(circuit.num_qubits, do_swaps=True, inverse=False), circuit.qubits)
circuit.save_statevector()
print(circuit.measure_all())
print(circuit.draw(output='text'))

simulator = AerSimulator()
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the aer simulator
job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("Counts:", counts)
for i, v in enumerate(result.get_statevector()):
    print(i, np.round(v, 2))
