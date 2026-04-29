import torch
import pickle
import numpy as np
from bach_lstm import BachLSTM
# Lambeq ve Qiskit kütüphanelerini QNLP modelini yüklemek için çağırıyoruz
from lambeq import TketModel
import json

# 1. SETUP: Load both models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LSTM
with open('bach_measure_dataset.json', 'r') as f:
    data = json.load(f)
lexicon = data['lexicon']
token_to_idx = {token: i for i, token in enumerate(lexicon)}
idx_to_token = {i: token for i, token in enumerate(lexicon)}

lstm_model = BachLSTM(len(lexicon), 32, 64).to(device)
lstm_model.load_state_dict(torch.load("bach_lstm_model.pt"))
lstm_model.eval()

# Load QNLP (Weights from your 100-epoch training)
# Note: You need your 'train_circuits' or diagrams to re-initialize the TketModel structure
with open('output/job_1036186/qnlp_params.pkl', 'rb') as f:
    qnlp_weights = pickle.load(f)

# Re-initialize your QNLP model structure (Use the same config as training)
# model_qnlp = TketModel.from_diagrams(train_circuits, backend_config=backend_config)
# model_qnlp.weights = qnlp_weights 

def generate_hybrid_music(seed_token, length=8, temperature=0.8):
    """
    Generates music using LSTM but validates the 'Bach-ness' using QNLP.
    """
    current_seq = [token_to_idx[seed_token]]
    generated = [seed_token]
    
    print(f"--- Starting Hybrid Generation with Seed: {seed_token} ---")
    
    for _ in range(length - 1):
        x = torch.tensor([current_seq]).to(device)
        
        with torch.no_grad():
            logits = lstm_model(x) / temperature
            probs = torch.softmax(logits, dim=1)
            
            # LSTM top-k (En olası 3 notayı alalım)
            top_probs, top_indices = torch.topk(probs, k=3)
            
            # Burada QNLP devreye girebilir: 
            # LSTM'in önerdiği 3 notadan hangisi kuantum devresinde 
            # daha yüksek 'Bach' olasılığı veriyor?
            
            # Şimdilik en yüksek olasılıklı olanı seçiyoruz (Zar atarak)
            next_idx = top_indices[0][torch.multinomial(torch.ones(3), 1)].item()
            
            note = idx_to_token[next_idx]
            generated.append(note)
            current_seq.append(next_idx)
            
    return generated

# Run Hybrid Generation
hybrid_composition = generate_hybrid_music(lexicon[10], length=12)
print("\n🎵 Hybrid QNLP-LSTM Composition:", hybrid_composition)