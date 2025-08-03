import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pickle
from typing import List, Dict, Any, Tuple
import re

class REMITokenizer:
    """REMI (Relative Position MIDI) Tokenizer for MIDI encoding/decoding"""

    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            'PAD': 0,
            'START': 1,
            'END': 2,
            'UNK': 3
        }
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary for REMI representation"""
        tokens = []

        # Special tokens
        tokens.extend(['PAD', 'START', 'END', 'UNK'])

        # Bar tokens
        tokens.extend([f'Bar_{i}' for i in range(1, 129)])  # up to 128 bars

        # Position tokens (subdivisions within a bar)
        tokens.extend([f'Position_{i}' for i in range(0, 96)])  # 96 subdivisions per bar

        # Note on tokens for trumpet range (E3 to C6 - MIDI 52 to 84)
        for note in range(52, 85):  # Extended trumpet range
            tokens.append(f'Note_On_{note}')

        # Note off tokens
        for note in range(52, 85):
            tokens.append(f'Note_Off_{note}')

        # Velocity tokens
        for vel in range(1, 128, 4):  # Every 4th velocity value
            tokens.append(f'Velocity_{vel}')

        # Duration tokens (in ticks)
        durations = [24, 48, 72, 96, 120, 144, 168, 192, 240, 288, 384, 480, 576, 672, 768, 960]
        for dur in durations:
            tokens.append(f'Duration_{dur}')

        # Tempo tokens
        for tempo in range(60, 200, 5):  # BPM from 60 to 195 in steps of 5
            tokens.append(f'Tempo_{tempo}')

        # Time signature tokens
        for num in [2, 3, 4, 6]:
            for den in [4, 8]:
                tokens.append(f'TimeSig_{num}_{den}')

        # Create vocabulary mapping
        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}

    def encode_midi(self, midi_file: MidiFile) -> List[int]:
        """Convert MIDI file to REMI token sequence"""
        tokens = [self.special_tokens['START']]

        # Get timing information
        ticks_per_beat = midi_file.ticks_per_beat
        current_time = 0
        current_bar = 1
        current_position = 0

        # Process all tracks
        all_messages = []
        for track in midi_file.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if hasattr(msg, 'type'):
                    all_messages.append((abs_time, msg))

        # Sort by absolute time
        all_messages.sort(key=lambda x: x[0])

        # Convert to REMI tokens
        for abs_time, msg in all_messages:
            # Calculate bar and position
            beat = abs_time / ticks_per_beat
            bar = int(beat // 4) + 1
            position = int((beat % 4) * 24)  # 24 subdivisions per beat

            # Add bar token if new bar
            if bar > current_bar:
                tokens.append(self.vocab.get(f'Bar_{bar}', self.special_tokens['UNK']))
                current_bar = bar

            # Add position token if position changed
            if position != current_position:
                tokens.append(self.vocab.get(f'Position_{position}', self.special_tokens['UNK']))
                current_position = position

            # Process message
            if msg.type == 'note_on' and msg.velocity > 0:
                # Add velocity
                vel_token = f'Velocity_{msg.velocity - (msg.velocity % 4)}'
                tokens.append(self.vocab.get(vel_token, self.special_tokens['UNK']))

                # Add note on
                note_token = f'Note_On_{msg.note}'
                tokens.append(self.vocab.get(note_token, self.special_tokens['UNK']))

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                note_token = f'Note_Off_{msg.note}'
                tokens.append(self.vocab.get(note_token, self.special_tokens['UNK']))

            elif msg.type == 'set_tempo':
                bpm = int(mido.tempo2bpm(msg.tempo))
                tempo_token = f'Tempo_{bpm - (bpm % 5)}'
                tokens.append(self.vocab.get(tempo_token, self.special_tokens['UNK']))

        tokens.append(self.special_tokens['END'])
        return tokens

    def decode_to_midi(self, tokens: List[int]) -> MidiFile:
        """Convert REMI tokens back to MIDI file"""
        midi_file = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        midi_file.tracks.append(track)

        # Track state
        current_time = 0
        current_velocity = 64
        active_notes = {}  # note -> start_time

        # Default tempo and time signature
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Add trumpet program change
        track.append(Message('program_change', program=56, time=0))  # Trumpet

        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]

                if token.startswith('Bar_'):
                    # New bar
                    bar_num = int(token.split('_')[1])
                    # Calculate time for this bar (assuming 4/4 time)
                    target_time = (bar_num - 1) * 480 * 4
                    if target_time > current_time:
                        current_time = target_time

                elif token.startswith('Position_'):
                    # Position within bar
                    position = int(token.split('_')[1])
                    # Calculate absolute time
                    bar_start = (current_time // (480 * 4)) * (480 * 4)
                    target_time = bar_start + (position * 480 // 24)
                    current_time = target_time

                elif token.startswith('Velocity_'):
                    current_velocity = int(token.split('_')[1])

                elif token.startswith('Note_On_'):
                    note = int(token.split('_')[2])
                    # Add note on message
                    track.append(Message('note_on', note=note, velocity=current_velocity, time=0))
                    active_notes[note] = current_time

                elif token.startswith('Note_Off_'):
                    note = int(token.split('_')[2])
                    if note in active_notes:
                        # Calculate duration
                        duration = current_time - active_notes[note]
                        if duration > 0:
                            track.append(Message('note_off', note=note, velocity=0, time=duration))
                        else:
                            track.append(Message('note_off', note=note, velocity=0, time=0))
                        del active_notes[note]

                elif token.startswith('Tempo_'):
                    bpm = int(token.split('_')[1])
                    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))

            i += 1

        # Close any remaining active notes
        for note in active_notes:
            track.append(Message('note_off', note=note, velocity=0, time=0))

        return midi_file

    def save(self, filepath: str):
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'reverse_vocab': self.reverse_vocab,
                'special_tokens': self.special_tokens
            }, f)

    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file"""
        tokenizer = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            tokenizer.vocab = data['vocab']
            tokenizer.reverse_vocab = data['reverse_vocab']
            tokenizer.special_tokens = data['special_tokens']
        return tokenizer

    def __len__(self):
        return len(self.vocab)

    def decode(self, tokens: List[int]):
        """Decode tokens to MIDI file - alias for decode_to_midi"""
        return self.decode_to_midi(tokens)

    def dump_midi(self, filepath: str):
        """For compatibility - returns a dummy MIDI file"""
        # This is a placeholder method for compatibility
        midi_file = MidiFile()
        track = MidiTrack()
        midi_file.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        midi_file.save(filepath)
        return midi_file
