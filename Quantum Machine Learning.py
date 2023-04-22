import qiskit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit_machine_learning.datasets import load_4D_heart_failure_data
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.utils import split_dataset_to_data_and_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
data, labels = load_4D_heart_failure_data()
data = StandardScaler().fit_transform(data)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
feature_map = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', reps=3)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))
optimizers = [COBYLA(maxiter=100), SPSA(maxiter=100), ADAM(maxiter=100)]
optimizer_names = ['COBYLA', 'SPSA', 'ADAM']
accuracy_scores = []
for i, optimizer in enumerate(optimizers):
    print(f"Training with {optimizer_names[i]} optimizer...")
    vqc = VQC(optimizer=optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel)
 vqc.fit(train_data, train_labels)
 predicted_labels = vqc.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted_labels)
    accuracy_scores.append(accuracy)
    print(f"Accuracy with {optimizer_names[i]} optimizer: {accuracy}")
plt.figure(figsize=(8, 6))
plt.bar(optimizer_names, accuracy_scores)
plt.xlabel('Optimizer')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Optimizers')
plt.show()
best_optimizer_index = np.argmax(accuracy_scores)
best_optimizer = optimizers[best_optimizer_index]
vqc = VQC(optimizer=best_optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel)
vqc.fit(train_data, train_labels)
predicted_labels = vqc.predict(test_data)
accuracy = accuracy_score(test_labels, predicted_labels)
confusion_mat = confusion_matrix(test_labels, predicted_labels)
classification_rep = classification_report(test_labels, predicted_labels)
fpr, tpr, _ = roc_curve(test_labels, predicted_labels)
roc_auc = auc(fpr, tpr)
print("Best Optimizer: ", optimizer_names[best_optimizer_index])
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion_mat)
print("Classification Report: \n", classification_rep)
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[
[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
feature_importances = vqc.get_feature_importance(train_data)
feature_names = [f'Feature {i+1}' for i in range(len(feature_importances))]
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues_r')
plt.title('Feature Importance')
plt.show()
reps_values = range(2, 11)
reps_accuracy_scores = []
for reps in reps_values:
    print(f"Training with reps = {reps}...")
    feature_map = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', reps=reps)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))
    vqc = VQC(optimizer=best_optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel)
 vqc.fit(train_data, train_labels)
 predicted_labels = vqc.predict(test_data)
 accuracy = accuracy_score(test_labels, predicted_labels)
    reps_accuracy_scores.append(accuracy)
    print(f"Accuracy with reps = {reps}: {accuracy}")
plt.figure(figsize=(8, 6))
plt.plot(reps_values, reps_accuracy_scores, marker='o')
plt.xlabel('Repetitions (reps)')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Repetitions (reps) in the Feature Map')
plt.show()
entanglement_blocks_values = ['cx', 'cz', 'crx', 'cry', 'crz']
entanglement_blocks_accuracy_scores = []
for entanglement_blocks in entanglement_blocks_values:
    print(f"Training with entanglement_blocks = {entanglement_blocks}...")
    feature_map = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks=entanglement_blocks, reps=3)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))
    vqc = VQC(optimizer=best_optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel)
 vqc.fit(train_data, train_labels)
predicted_labels = vqc.predict(test_data)
   accuracy= accuracy_score(test_labels, predicted_labels)
entanglement_blocks_accuracy_scores.append(accuracy)
print(f"Accuracy with entanglement_blocks = {entanglement_blocks}: {accuracy}")
  plt.figure(figsize=(8, 6))
plt.bar(entanglement_blocks_values, entanglement_blocks_accuracy_scores)
plt.xlabel('Entanglement Blocks')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Entanglement Blocks in the Feature Map')
plt.show()
rotation_blocks_values = [['ry'], ['rz'], ['rx'], ['ry', 'rz'], ['rx', 'ry'], ['rx', 'rz']]
rotation_blocks_labels = ['ry', 'rz', 'rx', 'ry, rz', 'rx, ry', 'rx, rz']
rotation_blocks_accuracy_scores = []
for rotation_blocks in rotation_blocks_values:
print(f"Training with rotation_blocks = {rotation_blocks}...")
feature_map = TwoLocal(rotation_blocks=rotation_blocks, entanglement_blocks='cx', reps=3)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))
vqc = VQC(optimizer=best_optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel)
vqc.fit(train_data, train_labels)
predicted_labels = vqc.predict(test_data)
accuracy = accuracy_score(test_labels, predicted_labels)
rotation_blocks_accuracy_scores.append(accuracy)
print(f"Accuracy with rotation_blocks = {rotation_blocks}: {accuracy}")
plt.figure(figsize=(8, 6))
plt.bar(rotation_blocks_labels, rotation_blocks_accuracy_scores)
plt.xlabel('Rotation Blocks')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Rotation Blocks in the Feature Map')
plt.show()
from qiskit_machine_learning.models import save_model, load_model
save_model(vqc, 'best_vqc_model.joblib')
loaded_vqc = load_model('best_vqc_model.joblib')
predicted_labels_loaded = loaded_vqc.predict(test_data)
assert np.array_equal(predicted_labels, predicted_labels_loaded), "Loaded model does not produce the same results"
print("Loaded model successfully tested.")
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
kf = KFold(n_splits=5)
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
for train_index, test_index in kf.split(data):
train_data_cv, test_data_cv = data[train_index], data[test_index]
train_labels_cv, test_labels_cv = labels[train_index], labels[test_index]
feature_map_cv = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', reps=3)
quantum_kernel_cv = QuantumKernel(feature_map=feature_map_cv, quantum_instance=Aer.get_backend('statevector_simulator'))
vqc_cv = VQC(optimizer=best_optimizer, feature_map=feature_map_cv, quantum_kernel=quantum_kernel_cv)
vqc_cv.fit(train_data_cv, train_labels_cv)
predicted_labels_cv = vqc_cv.predict(test_data_cv)
accuracy_cv = accuracy_score(test_labels_cv, predicted_labels_cv)
f1_cv = f1_score(test_labels_cv, predicted_labels_cv)
precision_cv = precision_score(test_labels_cv, predicted_labels_cv)
recall_cv = recall_score(test_labels_cv, predicted_labels_cv)
accuracy_scores.append(accuracy_cv)
f1_scores.append(f1_cv)
precision_scores.append(precision_cv)
recall_scores.append(recall_cv)
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
print("Cross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
optimizers = [COBYLA(maxiter=100), SPSA(maxiter=100), L_BFGS_B(maxiter=100)]
optimizer_names = ['COBYLA', 'SPSA', 'L_BFGS_B']
optimizer_performance = []
for opt, opt_name in zip(optimizers, optimizer_names):
print(f"Training with optimizer: {opt_name}")
vqc_opt = VQC(optimizer=opt, feature_map=feature_map, quantum_kernel=quantum_kernel)
# Train the quantum classifier
vqc_opt.fit(train_data, train_labels)
predicted_labels_opt = vqc_opt.predict(test_data)
accuracy_opt = accuracy_score(test_labels, predicted_labels_opt)
optimizer_performance.append(accuracy_opt)
print(f"Accuracy with optimizer {opt_name}: {accuracy_opt}")
plt.figure(figsize=(8, 6))
plt.bar(optimizer_names, optimizer_performance)
plt.xlabel('Optimizer')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Optimizers')
plt.show()
reps_values = list(range(1, 11))
reps_accuracy_scores = []
for reps in reps_values:
print(f"Training with reps = {reps}...")
feature_map_reps = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', reps=reps)
quantum_kernel_reps = QuantumKernel(feature_map=feature_map_reps, quantum_instance=Aer.get_backend('statevector_simulator'))
vqc_reps = VQC(optimizer=best_optimizer, feature_map=feature_map_reps, quantum_kernel=quantum_kernel_reps)
vqc_reps.fit(train_data, train_labels)
predicted_labels_reps = vqc_reps.predict(test_data)
accuracy_reps = accuracy_score(test_labels, predicted_labels_reps)
reps_accuracy_scores.append(accuracy_reps)
print(f"Accuracy with reps = {reps}: {accuracy_reps}")
plt.figure(figsize=(8, 6))
plt.plot(reps_values, reps_accuracy_scores)
plt.xlabel('Reps')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Reps in the Feature Map')
plt.show()
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 5 and not b.configuration().simulator))
print(f"Using backend: {backend}")
vqc_ibmq.fit(train_data, train_labels)
predicted_labels_ibmq = vqc_ibmq.predict(test_dataaccuracy_ibmq = accuracy_score(test_labels, predicted_labels_ibmq)
confusion_mat_ibmq = confusion_matrix(test_labels, predicted_labels_ibmq)
print("Accuracy on IBMQ backend: ", accuracy_ibmq)
print("Confusion Matrix on IBMQ backend: \n", confusion_mat_ibmq)
best_model_params = vqc.best_params
import json
with open('best_model_params.json', 'w') as f:
json.dump(best_model_params, f)
with open('best_model_params.json', 'r') as f:
loaded_best_model_params = json.load(f)
vqc_loaded_params = VQC(optimizer=best_optimizer, feature_map=feature_map, quantum_kernel=quantum_kernel, **loaded_best_model_params)
vqc_loaded_params.fit(train_data, train_labels)
predicted_labels_loaded_params = vqc_loaded_params.predict(test_data)
assert np.array_equal(predicted_labels, predicted_labels_loaded_params), "Loaded parameters do not produce the same results"
print("VQC with loaded parameters successfully tested.")
print("Summary of the final model:")
print(f"Feature Map: {feature_map}")
print(f"Quantum Kernel: {quantum_kernel}")
print(f"Optimizer: {best_optimizer}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix: \n{confusion_mat}")
print("Completed all experiments and evaluations.")
