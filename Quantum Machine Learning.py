import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit_machine_learning.datasets import load_4D_heart_failure_data
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils import split_dataset_to_data_and_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and preprocess the dataset
data, labels = load_4D_heart_failure_data()
data = StandardScaler().fit_transform(data)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Create a quantum feature map
feature_map = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', reps=3)

# Create a quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))

# Create a quantum classifier
optimizer = COBYLA(maxiter=100)
vqc = VQC(optimizer=optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel)

# Train the quantum classifier
vqc.fit(train_data, train_labels)

# Test the quantum classifier
predicted_labels = vqc.predict(test_data)

# Evaluate the results
accuracy = accuracy_score(test_labels, predicted_labels)
confusion_mat = confusion_matrix(test_labels, predicted_labels)

print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion_mat)
