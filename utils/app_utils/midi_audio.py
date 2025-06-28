import os
import io
import tempfile
import pretty_midi
import musicpy as mp

import numpy as np
from scipy.io import wavfile

import streamlit as st

from utils.app_utils.startapp import check_pygame_compatibility





def get_all_midis(folder_path):
    filenames = []

    # Check if the path exists
    if not os.path.exists(folder_path):
        raise Exception('Folder path for midi not found')

    files = os.listdir(folder_path)
    for file in files:
        filenames.append(file)

    return filenames



def load_midi_as_bytes(midi):
    if isinstance(midi, (str, bytes, os.PathLike)):
        midi_data = pretty_midi.PrettyMIDI(midi)
    else:
        midi_data = pretty_midi.PrettyMIDI(midi)

    # apply fluidsynth to midi data // fluidsynth is a pip library AND a package of its own. You need `package.txt`
    audio_data = midi_data.fluidsynth()
    audio_data = np.int16(
        audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
    )  # Normalize for 16-bit audio

    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, 44100, audio_data)
    virtualfile.seek(0)
    return virtualfile



def export_to_midi_as_bytes(piece):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_midi:
        midi_file_path = tmp_midi.name
        mp.write(piece, name=midi_file_path)

    with open(midi_file_path, "rb") as f:
        midi_bytes = f.read()

    os.remove(midi_file_path)
    return midi_bytes



def transcribe_piece_to_wav(piece):
    # Save the mp.chord or mp.track object as a MIDI file using MusicPy
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_midi:
        midi_file_path = tmp_midi.name
        mp.write(piece, name=midi_file_path)

    # Load the MIDI data as bytes and convert to audio
    virtualfile = load_midi_as_bytes(midi_file_path)

    # Delete the temporary MIDI file
    os.remove(midi_file_path)
    return virtualfile


    

def play_audio(audio_data, is_file=False, bpm=120):
    """ 
    Plays sound or returns an audio obj
    """
    if 'pygame_compatible' not in st.session_state:
        st.session_state['pygame_compatible'] = check_pygame_compatibility()

    if st.session_state['pygame_compatible']:
        mp.play(audio_data, bpm=bpm)
        return None
        
    else:
        if is_file: 
            audio = load_midi_as_bytes(audio_data)
        else:
            audio = transcribe_piece_to_wav(audio_data)
        return audio
    


@st.experimental_fragment
def download_midi_no_refresh(midi_fname, midi_bytes):
    """ 
    st.download_button() refreshes the page. The workaround is to wrap it in a fragment. 
    It is possible for the first download to fail.
    """
    st.download_button(
        label='Download MIDI', 
        data=midi_bytes, 
        file_name=midi_fname, 
        mime='audio/midi'
    )