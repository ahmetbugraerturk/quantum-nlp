import os
import sys
import json
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- HPC OPTIMIZATION: Force GPU and Statevector method at the OS level ---
os.environ['QISKIT_AER_DEVICE'] = 'GPU'
os.environ['QISKIT_AER_METHOD'] = 'statevector'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

from lambeq.backend.grammar import Id, Box, Word
from lambeq import AtomicType, IQPAnsatz, Dataset, QuantumTrainer, SPSAOptimizer, TketModel
from qiskit_aer import AerSimulator
from pytket.extensions.qiskit import AerBackend

# 1. Argument Parser for Cluster Execution
parser = argparse.ArgumentParser(description='QNLP Training for Bach Music Measures')
parser.add_argument('--data_path', type=str, default='bach_measure_dataset.json')
parser.add_argument('--epochs', type=int, default=20) 
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--batch_size', type=int, default=10)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 2. Backend Initialization (Manual Injection to bypass pytket version bugs)
try:
    # Use AerSimulator directly to ensure GPU and Statevector are active
    # We use 'single' precision because it's significantly faster on most GPUs
    gpu_sim = AerSimulator(device='GPU', method='statevector', precision='single')
    
    # Initialize AerBackend and manually overwrite the internal qiskit backend
    backend = AerBackend()
    backend._qiskit_backend = gpu_sim 
    
    print(f"--- QNLP Status: High-Performance GPU Backend Active ({gpu_sim.name}) ---", flush=True)
except Exception as e:
    print(f"--- GPU Setup Error: {e}. Falling back to default CPU. ---", flush=True)
    backend = AerBackend()

# 3. Data Preparation and Diagram Composition
def prepare_quantum_data(json_path, max_samples=50):
    """
    Parses the Bach measure dataset and converts music notes into quantum diagrams.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sentences = data['sentences'][:max_samples]
    n, s = AtomicType.NOUN, AtomicType.SENTENCE   
    
    diagrams, labels = [], []
    
    for i, sent in enumerate(sentences):
        try:
            if not sent: continue
            truncated_sent = sent[:5]
            
            words = [Word(str(note), n) for note in truncated_sent]
            diag = Id().tensor(*words)
            
            # Compositional box to transform Noun types into a Sentence type
            composer = Box('BACH_COMP', n ** len(truncated_sent), s)
            full_diag = diag >> composer
            
            diagrams.append(full_diag)
            # Binary label: 1 if A4 is in the measure, 0 otherwise
            labels.append([1, 0] if 'A4' in str(sent) else [0, 1])
            
        except Exception as e:
            print(f"--- Error processing sample {i}: {e} ---", flush=True)
            continue

    return diagrams, np.array(labels), n, s

train_diagrams, y_train, n, s = prepare_quantum_data(args.data_path)

# 4. Ansatz Definition (Mapping diagrams to 11-qubit circuits)
ansatz = IQPAnsatz({n: 2, s: 1}, n_layers=1)
train_circuits = [ansatz(d) for d in train_diagrams]

print(f"--- Circuit Compilation Finished. Total Qubits: {train_circuits[0].to_tk().n_qubits} ---", flush=True)

# 5. Hybrid Trainer Setup (Optimized for GPU speed)
loss = lambda y_hat, y: -np.sum(y * np.log(y_hat + 1e-9)) / len(y)
acc = lambda y_hat, y: np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))

backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2), 
    # CRITICAL: shots=None forces exact statevector simulation, 
    # which removes sampling overhead and lets the GPU fly!
    'shots': None 
}

model = TketModel.from_diagrams(train_circuits, backend_config=backend_config)

trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=args.epochs,
    optimizer=SPSAOptimizer,
    # Adjusted 'a' and 'c' for faster convergence in Bach measure classification
    optim_hyperparams={'a': 0.02, 'c': 0.04, 'A': 0.01, 'iterations': 1},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    log_dir=args.output_dir,
    verbose='text',
    seed=42
)

# 6. Training Loop (with 80-20 Train/Val Split)
train_circs, val_circs, y_train_split, y_val_split = train_test_split(
    train_circuits, y_train, test_size=0.2, random_state=42
)

train_dataset = Dataset(train_circs, y_train_split, batch_size=args.batch_size)
val_dataset = Dataset(val_circs, y_val_split, batch_size=args.batch_size)

print("--- Starting Hybrid Quantum Optimization ---", flush=True)
trainer.fit(train_dataset, val_dataset=val_dataset)

# 7. Model Persistence
params_path = os.path.join(args.output_dir, 'qnlp_bach_weights.pkl')
with open(params_path, 'wb') as f:
    pickle.dump(model.weights, f)

best_lt_path = os.path.join(args.output_dir, 'best_model.lt')

try:
    if os.path.exists(best_lt_path):
        # Madem torch "magic number" hatası verdi, direkt pickle ile dalıyoruz
        with open(best_lt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Checkpoint bir dict ise içinden weights'i alalım
        # Lambeq genelde {'model_weights': [...], 'epoch': ...} şeklinde tutar
        if isinstance(checkpoint, dict) and 'model_weights' in checkpoint:
            best_weights = checkpoint['model_weights']
            epoch = checkpoint.get('epoch', 'unknown')
        else:
            # Eğer direkt ağırlık listesiyse kendisini alalım
            best_weights = checkpoint
            epoch = "final/best"

        # Golden PKL dosyamızı kaydediyoruz
        best_pkl_path = os.path.join(args.output_dir, 'qnlp_best_weights.pkl')
        with open(best_pkl_path, 'wb') as f:
            pickle.dump(best_weights, f)
            
        print(f"--- [SUCCESS] Best weights extracted from {best_lt_path} (Epoch: {epoch}) ---", flush=True)
    else:
        print(f"--- [ERROR] {best_lt_path} not found! ---", flush=True)

except Exception as e:
    print(f"--- [EXTRACTION FAILED] Error: {e} ---", flush=True)

# 8. Metric Visualization for Research Report
print("\n--- Generating Metrics Plots ---", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Evolution
ax1.plot(trainer.train_costs, label='QNLP Train Loss', color='darkred', lw=2)
ax1.plot(trainer.val_costs, label='QNLP Val Loss', color='orange', linestyle='--')
ax1.set_title('Quantum SPSA Loss History')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Accuracy Evolution
if 'acc' in trainer.train_eval_results:
    ax2.plot(trainer.train_eval_results['acc'], label='Train Acc', color='navy', lw=2)
    ax2.plot(trainer.val_eval_results['acc'], label='Val Acc', color='blue', linestyle='--')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plot_path = os.path.join(args.output_dir, 'qnlp_performance_metrics.png')
plt.tight_layout()
plt.savefig(plot_path, dpi=300)

print(f"--- SUCCESS: Optimized Weights saved to {params_path} ---", flush=True)
print(f"--- SUCCESS: Performance Plots saved to {plot_path} ---", flush=True)