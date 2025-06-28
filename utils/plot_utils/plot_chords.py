import mido
import pandas as pd
import plotly.graph_objects as go
import musicpy as mp


def parse_midi(file_path, start_time=None, end_time=None):
    mid = mido.MidiFile(file_path)
    note_events = []
    track_time_counters = [0] * len(mid.tracks)  # Track individual time counters

    for i, track in enumerate(mid.tracks):
        for msg in track:
            track_time_counters[i] += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                note_events.append({
                    'track': i,
                    'timestamp': mido.tick2second(track_time_counters[i], mid.ticks_per_beat, 500000),
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'type': msg.type,
                    'length': mido.tick2second(msg.time, mid.ticks_per_beat, 500000) if msg.type == 'note_on' else 0
                })

    df = pd.DataFrame(note_events)
    if start_time:
        df = df[df['timestamp'] >= start_time]
    if end_time:
        df = df[df['timestamp'] <= end_time]

    return df

def plot_midi_notes(df: pd.DataFrame, time_signature=4, height=500, width=None, title=None):
    fig = go.Figure()

    # Define the range of MIDI notes to display
    min_note = 10
    max_note = 98
    max_time = int(df['timestamp'].max())
    df = df[df['type'] == 'note_on']

    # Shading for black keys
    black_keys = [1, 3, 6, 8, 10]
    for note in range(min_note, max_note + 1):
        if note % 12 in black_keys:
            fig.add_shape(
                type="rect",
                x0=0,
                x1=max_time,
                y0=note - 0.5,
                y1=note + 0.5,
                fillcolor="#6A687E",
                opacity=0.5,
                layer="below"
            )

    # Shading for bars
    bar_length = 1
    shade_interval = bar_length * time_signature
    for start_time in range(0, max_time + shade_interval, shade_interval * 2):

        # add truncation rule
        end_time = start_time + shade_interval
        if end_time > max_time:
            end_time = max_time

        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=start_time,
            x1=end_time,
            y0=0,
            y1=1,
            fillcolor="#71aa91",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

    # Plot each track with a different color and lines for note duration
    for track in df['track'].unique():
        track_data = df[df['track'] == track]
        fig.add_trace(go.Scatter(
            x=track_data['timestamp'],
            y=track_data['note'],
            mode='markers',
            name=f'Track {track}'
        ))

    tickvals = [11.5, 23.5, 35.5, 47.5, 59.5, 71.5, 83.5, 95.5]
    ticktext = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        
    fig.update_layout(
        height=height, 
        width=width,
        title=title if title != None else 'MIDI Note Events',
        xaxis_title='Time (bar)',
        yaxis_title='MIDI Note Number',
        xaxis=dict(
            dtick=time_signature,  # Set gridline interval to time, default 4
            showgrid=True,
            showticklabels=False,
        ),
        yaxis=dict(
            tick0=11.5, # 10+0.5 for offset 
            dtick=12, # octave per tick
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
        ),
        template='simple_white',
    )
    return fig


def plot_chords(chord, start_time=0, end_time=10, time_signature=4, height=500, width=None, title=None):
    # generate temp midi file
    mp.write(chord, name='temp.mid')

    # Parse the generated MIDI file
    df = parse_midi('temp.mid', start_time, end_time)
    fig = plot_midi_notes(df, time_signature, height, width, title=title)
    return fig