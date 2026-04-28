import torch
import json
from bach_lstm import BachLSTM 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('bach_measure_dataset.json', 'r') as f:
    data = json.load(f)

lexicon = data['lexicon']
token_to_idx = {token: i for i, token in enumerate(lexicon)}
idx_to_token = {i: token for i, token in enumerate(lexicon)}

model = BachLSTM(len(lexicon), 32, 64).to(device)
model.load_state_dict(torch.load("bach_lstm_model.pt"))
model.eval()

def generate_music(seed_token, length=8):
    current_seq = [token_to_idx[seed_token]]
    generated = [seed_token]
    
    for _ in range(length - 1):
        x = torch.tensor([current_seq]).to(device)
        output = model(x)
        next_idx = torch.argmax(output, dim=1).item()
        
        generated.append(idx_to_token[next_idx])
        current_seq.append(next_idx)
    
    return generated

def generate_music_with_temperature(seed_token, length=8, temperature=1.0):
    model.eval()
    current_seq = [token_to_idx[seed_token]]
    generated = [seed_token]
    
    for _ in range(length - 1):
        x = torch.tensor([current_seq]).to(device)
        
        with torch.no_grad():
            logits = model(x) # Modelden çıkan ham sayılar (logits)
            
            # Sıcaklık uyguluyoruz
            logits = logits / temperature
            
            # Softmax ile olasılığa çeviriyoruz
            probs = torch.softmax(logits, dim=1)
            
            # Bu olasılık dağılımından bir tane örnek çekiyoruz (Zar atıyoruz)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(idx_to_token[next_idx])
            current_seq.append(next_idx)
            
    return generated

# Test edelim
# 0.8 genelde Bach için 'tatlı nokta'dır; ne çok rastgele ne çok sıkıcı.
sample_output = generate_music_with_temperature(lexicon[0], length=8, temperature=0.8)
print("Yeni Beste:", sample_output)

# Test: Rastgele bir Bach notasıyla başlat
# sample_output = generate_music(lexicon[0])
# print("Modelin Bestesi:", sample_output)