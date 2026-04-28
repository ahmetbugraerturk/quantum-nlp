import music21
from collections import Counter
import json

def prepare_measure_based_dataset(max_notes=6):
    # Search for Bach Chorales in the music21 corpus
    bach_bundle = music21.corpus.search('bach')
    print(f"Total Bach files found: {len(bach_bundle)}")

    all_sentences = []
    raw_tokens_for_lexicon = []

    for result in bach_bundle:
        # Filter for Chorales based on BWV range (250-438)
        # result.sourcePath can be a PosixPath; convert to string before using 'in' or split
        source_path_str = str(result.sourcePath)
        if "bwv" in source_path_str:
            try:
                bwv_num = float(source_path_str.split('bwv')[-1].split('.')[0])
                if not (250 <= bwv_num <= 438):
                    continue
            except Exception:
                continue

            score = result.parse()
            # Focus on the Soprano part (usually the top melody)
            part = score.parts[0]
            
            # Iterate through each measure of the piece
            for measure in part.getElementsByClass(music21.stream.Measure):
                # Filter: We only want 4/4 time signature measures (duration 4.0)
                if measure.duration.quarterLength != 4.0:
                    continue
                
                current_measure_notes = []
                notes_and_chords = measure.flatten().notes
                
                # Check the number of notes to keep the quantum circuit depth manageable
                if 1 <= len(notes_and_chords) <= max_notes:
                    for element in notes_and_chords:
                        if isinstance(element, music21.note.Note):
                            # Token format: Pitch_DurationType (e.g., C4_eighth)
                            token = f"{element.pitch.nameWithOctave}_{element.duration.type}"
                            current_measure_notes.append(token)
                        elif isinstance(element, music21.chord.Chord):
                            # Take the highest note for melodic consistency
                            highest = element.sortAscending().notes[-1]
                            token = f"{highest.pitch.nameWithOctave}_{element.duration.type}"
                            current_measure_notes.append(token)
                    
                    if current_measure_notes:
                        all_sentences.append(current_measure_notes)
                        raw_tokens_for_lexicon.extend(current_measure_notes)

    # 4. Create Lexicon: Top 30 most frequent note-duration combinations
    token_counts = Counter(raw_tokens_for_lexicon)
    lexicon = [t[0] for t in token_counts.most_common(30)]
    
    # 5. Final Filtering: Keep sentences where all notes are in the Lexicon
    final_sentences = [s for s in all_sentences if all(t in lexicon for t in s)]

    output = {
        "lexicon": lexicon,
        "sentences": final_sentences,
        "config": {
            "qubits_estimated": 12,
            "target_measure_duration": 4.0,
            "max_notes_per_measure": max_notes
        }
    }

    with open('bach_measure_dataset.json', 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Success! Created {len(final_sentences)} measure-based sentences.")
    return output

# Execute the preparation
prepare_measure_based_dataset(max_notes=6)