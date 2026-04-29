import pickle
import json
import torch
import numpy as np
import random
from lambeq.backend.grammar import Id, Box, Word
from lambeq import AtomicType, IQPAnsatz, TketModel
from qiskit_aer import AerSimulator
from pytket.extensions.qiskit import AerBackend

# --- 1. Load Lexicon and Settings ---
with open('bach_measure_dataset.json', 'r') as f:
    data = json.load(f)

lexicon = data['lexicon']
idx_to_token = {i: token for i, token in enumerate(lexicon)}
token_to_idx = {token: i for i, token in enumerate(lexicon)}

# Settings (Eğitimdekiyle birebir aynı olmalı)
WINDOW_SIZE = 3
N_LAYERS = 2
n, s = AtomicType.NOUN, AtomicType.SENTENCE

# --- 2. Load the Model Weights ---
WEIGHTS_PATH = 'output/job_1037047/generative_qnlp_weights.pkl'
with open(WEIGHTS_PATH, 'rb') as f:
    trained_weights = pickle.load(f)

# --- 3. Setup Backend (GPU) ---
# Üretim yaparken de GPU kullanmak hızı artırır
gpu_sim = AerSimulator(device='GPU', method='statevector')
backend = AerBackend()
backend._qiskit_backend = gpu_sim

ansatz = IQPAnsatz({n: 1, s: 1}, n_layers=N_LAYERS)

def generate_step(context_tokens, weights):
    """
    Takes 3 notes, builds a quantum circuit, and predicts the next note.
    """
    # 1. Create Diagram
    words = [Word(str(note), n) for note in context_tokens]
    diag = Id().tensor(*words) >> Box('COMP', n ** len(context_tokens), s)
    
    # 2. Create Circuit
    circuit = ansatz(diag)
    
    # 3. Setup Backend Config with MANDATORY compilation pass
    # Compilation pass seviyesini 2 yaparak GPU üzerinde hızlı bir optimizasyon sağlıyoruz.
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2), # EKSİK OLAN SATIR BUYDU!
        'shots': None 
    }
    
    # 4. Initialize TketModel
    model = TketModel.from_diagrams([circuit], backend_config=backend_config)
    
    # 5. Inject trained weights
    model.weights = weights
    
    # 6. Prediction
    probs = model([circuit])[0] 
    return probs

def composer_bot(seed_notes, length=8, temperature=0.8):
    generated = list(seed_notes)
    
    print(f"--- [START] Seed: {' - '.join(seed_notes)} ---")
    
    for i in range(length):
        context = generated[-WINDOW_SIZE:]
        
        # Kuantum modelinden olasılıkları al
        probs = generate_step(context, trained_weights)
        
        # Temperature Scaling (Sıcaklık Ayarı)
        # Logitler üzerinden değil doğrudan olasılıklar üzerinden basit bir scaling
        probs = np.exp(np.log(probs + 1e-9) / temperature)
        probs /= np.sum(probs)
        
        # Bir sonraki notayı seç (Lexicon'un ilk yarısı mı ikinci yarısı mı?)
        # Hatırla: Eğitimde [1,0] ve [0,1] olarak etiketlemiştik
        choice = np.random.choice([0, 1], p=probs)
        
        # Lexicon içinden uygun gruptan rastgele bir nota seçelim
        half = len(lexicon) // 2
        if choice == 0:
            next_note = np.random.choice(lexicon[:half])
        else:
            next_note = np.random.choice(lexicon[half:])
            
        generated.append(next_note)
        print(f"Step {i+1}: Added {next_note}")
        
    return generated

# --- RUN COMPOSER ---
# Başlangıç notaları (Seed) veri setinden rastgele 3 nota olabilir
rand = random.randint(0, len(data['sentences']) - 1)
seed = data['sentences'][rand][:WINDOW_SIZE]  # İlk cümlenin ilk 3 notasını seed olarak alıyoruz
new_bach_piece = composer_bot(seed, length=10, temperature=0.8)

print("\n--- Given Composition (first three notes) --- and --- FINAL COMPOSITION ---")
print("generate_wav_from_tokens([" + ", ".join(f'"{note}"' for note in data['sentences'][rand]) + "])")
print("generate_wav_from_tokens([" + ", ".join(f'"{note}"' for note in new_bach_piece) + "])")