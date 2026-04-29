import os
import json
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. HPC Optimization: Use your working GPU setup ---
os.environ['QISKIT_AER_DEVICE'] = 'GPU'
os.environ['QISKIT_AER_METHOD'] = 'statevector'

from lambeq.backend.grammar import Id, Box, Word
from lambeq import AtomicType, IQPAnsatz, Dataset, QuantumTrainer, SPSAOptimizer, TketModel
from qiskit_aer import AerSimulator
from pytket.extensions.qiskit import AerBackend

# Argument Parser
parser = argparse.ArgumentParser(description='QNLP Generative Training for Bach')
parser.add_argument('--data_path', type=str, default='bach_measure_dataset.json')
parser.add_argument('--epochs', type=int, default=50) 
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--window_size', type=int, default=3)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# --- 2. GPU Backend Initialization (The one you fixed!) ---
try:
    gpu_sim = AerSimulator(device='GPU', method='statevector')
    backend = AerBackend()
    backend._qiskit_backend = gpu_sim 
    print(f"--- [SUCCESS] Using fixed GPU Backend: {gpu_sim.name} ---", flush=True)
except Exception as e:
    print(f"--- [ERROR] GPU failed, check your setup: {e} ---", flush=True)
    exit(1)

# --- 3. Generative Data Preparation (Sliding Window) ---
def prepare_generative_data(json_path, window_size=3, max_samples=250):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lexicon = data['lexicon']
    sentences = data['sentences']
    token_to_idx = {token: i for i, token in enumerate(lexicon)}
    vocab_size = len(lexicon)
    
    n, s = AtomicType.NOUN, AtomicType.SENTENCE   
    diagrams, labels = [], []
    
    for sent in sentences:
        if len(sent) <= window_size: continue
        for i in range(len(sent) - window_size):
            context = sent[i : i + window_size]
            target_token = sent[i + window_size]
            
            if target_token not in token_to_idx: continue
            
            # Words as context
            words = [Word(str(note), n) for note in context]
            diag = Id().tensor(*words)
            
            # Compositional box to merge window into a sentence embedding
            composer = Box('COMP', n ** window_size, s)
            full_diag = diag >> composer
            
            diagrams.append(full_diag)
            
            # Target as One-Hot (SPSA will map output to these classes)
            # TketModel with s:1 produces 2 output probabilities. 
            # We map target to 0 or 1 for this simple generative task.
            target_idx = token_to_idx[target_token]
            # Simplifying: Is target in the first half of the lexicon or second?
            # (To keep output dimension compatible with s:1)
            labels.append([1, 0] if target_idx < vocab_size // 2 else [0, 1])
            
            if len(diagrams) >= max_samples: break
        if len(diagrams) >= max_samples: break

    return diagrams, np.array(labels), n, s

print("--- [DATA] Preparing sliding window diagrams ---", flush=True)
train_diagrams, y_train, n, s = prepare_generative_data(args.data_path, window_size=args.window_size)

# --- 4. Circuit Compilation ---
ansatz = IQPAnsatz({n: 1, s: 1}, n_layers=2)
train_circuits = [ansatz(d) for d in train_diagrams]

# --- 5. TketModel Setup (Sequence Logic) ---
# Cross-Entropy Loss for Next-Token probability mapping
loss = lambda y_hat, y: -np.sum(y * np.log(y_hat + 1e-9)) / len(y)
acc = lambda y_hat, y: np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))

backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2), 
    'shots': None # Exact statevector on GPU
}

model = TketModel.from_diagrams(train_circuits, backend_config=backend_config)

# --- 6. Hybrid Training with SPSA ---
trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=args.epochs,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.05, 'c': 0.06, 'A': 0.01, 'iterations': 1},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    log_dir=args.output_dir,
    verbose='text',
    seed=42
)

train_circs, val_circs, y_train_split, y_val_split = train_test_split(
    train_circuits, y_train, test_size=0.2, random_state=42
)

train_dataset = Dataset(train_circs, y_train_split, batch_size=8)
val_dataset = Dataset(val_circs, y_val_split, batch_size=8)

print("--- [TRAINING] Starting SPSA Optimization on GPU ---", flush=True)
trainer.fit(train_dataset, val_dataset=val_dataset)

# --- 7. Save and Visualize ---
params_path = os.path.join(args.output_dir, 'generative_qnlp_weights.pkl')
with open(params_path, 'wb') as f:
    pickle.dump(model.weights, f)

# --- Extracting Best Model Weights ---
best_lt_path = os.path.join(args.output_dir, 'best_model.lt')
best_pkl_path = os.path.join(args.output_dir, 'qnlp_best_weights.pkl')

try:
    if os.path.exists(best_lt_path):
        # Open the lambeq checkpoint (it's a pickle)
        with open(best_lt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Extract weights from the checkpoint dictionary
        if isinstance(checkpoint, dict) and 'model_weights' in checkpoint:
            best_weights = checkpoint['model_weights']
            best_epoch = checkpoint.get('epoch', 'unknown')
            print(f"--- [INFO] Found Best Model at Epoch {best_epoch} ---", flush=True)
        else:
            best_weights = checkpoint # Direct weight list fallback

        # Save as a clean PKL for your inference script
        with open(best_pkl_path, 'wb') as f:
            pickle.dump(best_weights, f)
            
        print(f"--- [SUCCESS] Best weights extracted to {best_pkl_path} ---", flush=True)
    else:
        print(f"--- [WARNING] {best_lt_path} not found. Using final weights instead. ---", flush=True)
        # Fallback to final weights if best is missing
        with open(best_pkl_path, 'wb') as f:
            pickle.dump(model.weights, f)
except Exception as e:
    print(f"--- [ERROR] Failed to extract best model: {e} ---", flush=True)

# --- Plotting Loss and Accuracy ---
print("\n--- [VISUALIZATION] Generating Training Metrics Plots ---", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss Evolution
ax1.plot(trainer.train_costs, label='Train Loss', color='darkred', lw=2)
ax1.plot(trainer.val_costs, label='Val Loss', color='orange', linestyle='--')
ax1.set_title('Quantum SPSA Loss History')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Accuracy Evolution
if 'acc' in trainer.train_eval_results:
    ax2.plot(trainer.train_eval_results['acc'], label='Train Acc', color='navy', lw=2)
    ax2.plot(trainer.val_eval_results['acc'], label='Val Acc', color='blue', linestyle='--')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plot_path = os.path.join(args.output_dir, 'qnlp_training_performance.png')
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"--- [SUCCESS] Plots saved to {plot_path} ---", flush=True)

print(f"--- [SUCCESS] Training complete. Weights saved to {params_path} ---", flush=True)