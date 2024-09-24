# This file will handle the quantum-enhanced optimization part of the project,
# especially for hyperparameter tuning using Grover's algorithm
# and the Shukla-Vedula algorithm. 

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator, MCMT
from qiskit.circuit.library.data_preparation import UniformSuperspositionGate
from qiskit.visualization import plot_histogram


def grover_oracle(marked_states, num_qubits):
    """
    Creates a Grover oracle that flips the phase of the marked states.

    Parameters:
    marked_states: List of binary strings representing the states to mark
    num_qubits: Number of qubits in the quantum circuit

    Returns:
    QuantumCircuit implementing the oracle
    """
    qc = QuantumCircuit(num_qubits)
    
    # Iterate over each target state in the list of marked states
    for target in marked_states:
        rev_target = target[::-1]  # Reverse the bit-string to match Qiskit’s qubit ordering
        
        zero_inds = [i for i, bit in enumerate(rev_target) if bit == '0']
        control_qubits = [i for i, bit in enumerate(rev_target) if bit == '1']
        
        qc.x(zero_inds)  # Apply X gates to qubits corresponding to '0' in the target state
        
        if control_qubits:
            qc.append(MCMT(gate='z', num_ctrl_qubits=len(control_qubits), num_target_qubits=1), control_qubits + [zero_inds[0]])

        qc.x(zero_inds)  # Reapply X gates to revert the initial state
    
    return qc

def diffusion_operator(qc, qubits):
    """
    Implements the diffusion operator in Grover's algorithm.

    Parameters:
    qc: QuantumCircuit to apply the diffusion operator to
    qubits: List of qubits to apply the diffusion operator to
    """
    qc.h(qubits)  # Apply Hadamard gates to all qubits
    qc.x(qubits)  # Apply Pauli-X gates to all qubits
    
    # Apply multi-controlled Z gate (phase flip if all qubits are |1>)
    qc.h(qubits[-1])  # Transform the last qubit to X basis
    qc.mcx(qubits[:-1], qubits[-1])  # Multi-controlled NOT gate
    qc.h(qubits[-1])  # Transform back from X basis
    
    qc.x(qubits)  # Apply Pauli-X gates again
    qc.h(qubits)  # Apply Hadamard gates again


def grover_algorithm_with_sv(num_qubits, marked_states, M):
    """
    Runs Grover's algorithm using Shukla-Vedula uniform superposition.

    Parameters:
    num_qubits: Number of qubits in the quantum circuit
    marked_states: List of binary strings representing the states to mark
    M: Number of computational basis states

    Returns:
    Results of the quantum circuit execution
    """
    # Create the quantum circuit
    qc = QuantumCircuit(num_qubits)

    # Apply the UniformSuperpositionGate from Shukla-Vedula algorithm
    usp_gate = UniformSuperspositionGate(M, num_qubits)
    qc.append(usp_gate, list(range(num_qubits)))

    # Create and append the oracle circuit
    oracle_circuit = grover_oracle(marked_states, num_qubits)
    qc.compose(oracle_circuit, inplace=True)
    
    # Calculate the optimal number of iterations
    optimal_num_iterations = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2**num_qubits))))
    
    # Apply Grover's operator the optimal number of times
    for _ in range(optimal_num_iterations):
        qc.compose(oracle_circuit, inplace=True)  # Apply the oracle
        diffusion_operator(qc, range(num_qubits))  # Apply the diffusion operator
    
    # Add measurements
    qc.measure_all()
    
    # Set up the simulator and run the circuit
    simulator = AerSimulator()
    transpiled_circuit = transpile(qc, simulator)
    result = simulator.run(transpiled_circuit, shots=1024).result()
    
    return result
