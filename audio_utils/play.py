import music21
from midi2audio import FluidSynth
import os

def generate_wav_from_tokens(token_list, output_name="bach_composition", soundfont_path="audio_utils/GeneralUser-GS/GeneralUser-GS.sf2"):
    """
    Takes a list of tokens and converts them directly to a .wav file.
    Example: ["C4_quarter", "G4_eighth", "A4_eighth", "B4_half"]
    """
    midi_file = f"{output_name}.mid"
    wav_file = f"{output_name}.wav"
    
    # --- STEP 1: TOKENS TO MIDI ---
    stream = music21.stream.Stream()
    for token in token_list:
        try:
            pitch_part, duration_part = token.split('_')
            n = music21.note.Note(pitch_part)
            n.duration.type = duration_part
            stream.append(n)
        except Exception as e:
            print(f"Skipping invalid token {token}: {e}")
            
    stream.write('midi', fp=midi_file)
    print(f"MIDI created: {midi_file}")

    # --- STEP 2: MIDI TO WAV (The Audio Generation) ---
    if os.path.exists(soundfont_path):
        fs = FluidSynth(soundfont_path)
        fs.midi_to_audio(midi_file, wav_file)
        print(f"SUCCESS! Audio file created: {wav_file}")
    else:
        print(f"ERROR: SoundFont file not found at {soundfont_path}")
        print("Please download a .sf2 file (e.g., GeneralUser GS) to generate WAV.")

    # --- ADIM 3: MIDI TEMİZLEME ---
    if os.path.exists(midi_file):
        os.remove(midi_file)
        print(f"Removed: {midi_file}")


# Test function to generate WAV files from random samples in the dataset
'''
import json
import random

def test_random_samples(json_path="bach_measure_dataset.json", num_samples=5):
    # JSON dosyasını yükle
    if not os.path.exists(json_path):
        print(f"Hata: {json_path} bulunamadı!")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sentences = data['sentences']
    
    # Rastgele 5 cümle seç
    sample_sentences = random.sample(sentences, min(num_samples, len(sentences)))
    
    print(f"Toplam {len(sentences)} cümle arasından {len(sample_sentences)} örnek seçildi.\n")

    for i, sentence in enumerate(sample_sentences):
        print(f"Örnek {i+1} işleniyor: {sentence}")
        output_filename = f"bach_test_sample_{i+1}"
        generate_wav_from_tokens(sentence, output_name=output_filename)
        print("-" * 30)

# Kullanım:
# Not: Aynı klasörde bir .sf2 dosyası olduğundan emin olun.
test_random_samples()
'''