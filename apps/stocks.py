# type: ignore
import streamlit as st
from streamlit import session_state as state
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import musicpy as mp
from utils.plot_utils.plot_stock import (
    make_linechart, 
    make_barchart,
)
from utils.plot_utils.plot_chords import (
    plot_chords,
)
from utils.parse_utils import parse_yf_df
from utils.app_utils.midi_audio import export_to_midi_as_bytes, play_audio
from utils.constants.ist_maps import POPULAR_INSTRUMENTS, RHYTHM_VARIANTS, PATTERN_VARIANTS

# Musical scales and their MIDI note mappings
MUSICAL_SCALES = {
    "C Major": [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84],  # C, D, E, F, G, A, B
    "C Minor": [60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84],  # C, D, Eb, F, G, Ab, Bb
    "C Minor Pentatonic": [60, 63, 65, 67, 70, 72, 75, 77, 79, 82, 84],  # C, Eb, F, G, Bb
    "C Major Pentatonic": [60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84],  # C, D, E, G, A
    "C Blues": [60, 63, 65, 66, 67, 70, 72, 75, 77, 78, 79, 82, 84],  # C, Eb, F, F#, G, Bb
    "C Dorian": [60, 62, 63, 65, 67, 69, 70, 72, 74, 75, 77, 79, 81, 82, 84],  # C, D, Eb, F, G, A, Bb
    "C Mixolydian": [60, 62, 64, 65, 67, 69, 70, 72, 74, 76, 77, 79, 81, 82, 84],  # C, D, E, F, G, A, Bb
    "C Lydian": [60, 62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84],  # C, D, E, F#, G, A, B
}

# Enhanced preset configurations
MUSICAL_PRESETS = {
    "Ambient": {
        "scale": "C Minor Pentatonic",
        "bpm": 80,
        "note_duration": 1.0,
        "pitch_source": "Close",
        "description": "Chill, atmospheric vibes with long, sustained notes"
    },
    "Electronic": {
        "scale": "C Minor",
        "bpm": 128,
        "note_duration": 0.25,
        "pitch_source": "Close",
        "description": "High-energy electronic with sharp, rhythmic patterns"
    },
    "Jazz": {
        "scale": "C Dorian",
        "bpm": 100,
        "note_duration": 0.5,
        "pitch_source": "High",
        "description": "Sophisticated jazz with swing and improvisation feel"
    },
    "Rock": {
        "scale": "C Mixolydian",
        "bpm": 140,
        "note_duration": 0.25,
        "pitch_source": "Close",
        "description": "High-energy rock with driving rhythms"
    },
    "Blues": {
        "scale": "C Blues",
        "bpm": 90,
        "note_duration": 0.5,
        "pitch_source": "Low",
        "description": "Soulful blues with expressive bends and slides"
    }
}

def get_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance with better error handling"""
    try:
        # Try with explicit timezone handling
        stock_data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            progress=False,
            ignore_tz=True,  # Ignore timezone issues
            prepost=False    # Don't include pre/post market data
        )
        
        if stock_data is None or stock_data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
            
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in stock_data.columns for col in required_cols):
            st.error(f"Missing required columns for {ticker}. Found: {list(stock_data.columns)}")
            return None
            
        return stock_data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        
        # Try alternative approach
        try:
            st.info("Trying alternative data source...")
            # Try with different parameters
            stock_data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                ignore_tz=True,
                prepost=False,
                threads=False  # Disable threading
            )
            
            if stock_data is not None and not stock_data.empty:
                st.success(f"✅ Successfully loaded data using alternative method")
                return stock_data
                
        except Exception as e2:
            st.error(f"Alternative method also failed: {str(e2)}")
        
        # Provide helpful suggestions
        st.warning("""
        **Troubleshooting tips:**
        - Check if the ticker symbol is correct (e.g., 'AAPL' not 'apple')
        - Try a different date range
        - Some stocks might be delisted or have restricted data
        - You can also upload your own CSV file instead
        """)
        
        return None

def parse_csv_data(uploaded_file):
    """Parse uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if DataFrame is empty
        if df is None or df.empty:
            st.error("CSV file is empty")
            return None
            
        # Check if it has the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return None
        
        # Try to parse date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif df.index.name is None:
            # Assume first column is date
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        return None

def normalize_data(data, min_val, max_val, target_min, target_max):
    """Normalize data to a target range"""
    if max_val == min_val:
        return target_min
    normalized = (data - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min

def create_midi_from_stock_data(stock_data, scale_name, bpm, note_duration=0.25, pitch_source="Close"):
    """Convert stock data to a musicpy chord with sequential timing"""
    if stock_data is None or stock_data.empty:
        return None

    scale = MUSICAL_SCALES.get(scale_name, MUSICAL_SCALES["C Minor Pentatonic"])
    if pitch_source == "Close":
        pitch_data = stock_data['Close']
    elif pitch_source == "High":
        pitch_data = stock_data['High']
    elif pitch_source == "Low":
        pitch_data = stock_data['Low']
    elif pitch_source == "Open":
        pitch_data = stock_data['Open']
    else:
        pitch_data = stock_data['Close']

    pitch_min, pitch_max = pitch_data.min(), pitch_data.max()

    # Build chord string with proper musicpy syntax
    chord_parts = []
    
    for index, row in stock_data.iterrows():
        price_normalized = normalize_data(pitch_data[index], pitch_min, pitch_max, 0, len(scale) - 1)
        scale_index = int(price_normalized)
        scale_index = max(0, min(scale_index, len(scale) - 1))
        note_pitch = scale[scale_index]
        note_name = mp.degree_to_note(note_pitch)
        
        # Format: {note}{pitch}[{duration;.]
        chord_parts.append(f'{note_name}[{note_duration};.]')

    # Join all notes into a chord string
    chord_string = ', '.join(chord_parts)
    piece = mp.chord(chord_string)
    
    return piece

def main():
    state['stock_data'] = state.get('stock_data', None)
    state['stock_song'] = state.get('stock_song', None)
    
    st.title('💰 Money to Music 🎵')
    st.markdown("Transform stock market data into musical compositions!")

    # Add session state display to sidebar for transparency
    with st.sidebar:
        st.markdown("### 🔧 Session State (Debug)")
        with st.expander("Current State", expanded=False):
            st.json({
                'stock_data_loaded': state['stock_data'] is not None,
                'stock_song_generated': state['stock_song'] is not None,
                'ticker': state.get('ticker', 'None'),
                'composition_details': state.get('composition_details', 'None')
            })
        
        st.markdown("### 📋 Preset Details")
        with st.expander("Available Presets", expanded=False):
            for preset_name, preset_data in MUSICAL_PRESETS.items():
                st.markdown(f"**{preset_name}:**")
                st.json(preset_data)

    with st.expander('Help', expanded=False):
        confused()

    # Step 1: Data Input
    with st.expander('📈 Step 1: Get Stock Data', expanded=True):
        data_input_form()

    # Step 2: Data Visualization
    if state['stock_data'] is not None:
        with st.expander('📊 Step 2: Review Data', expanded=True):
            data_visualization()
        
        # Step 3: Music Generation
        with st.expander('🎵 Step 3: Generate Music', expanded=True):
            music_generation_form()

    # Step 4: Results
    if state['stock_song'] is not None:
        with st.expander('🎼 Step 4: Download & Play', expanded=True):
            results_section()

def generate_sample_data(days=30):
    """Generate sample stock data for testing"""
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic stock data
    base_price = 100
    price_values = [base_price]  # Track actual price values
    data_rows = []
    
    for i in range(days):
        # Random walk for price
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 2)  # Daily change with std dev of 2
            price = price_values[-1] * (1 + change/100)
        
        # Generate OHLC from base price
        daily_volatility = np.random.uniform(0.5, 2.0)
        open_price = price * (1 + np.random.uniform(-daily_volatility/100, daily_volatility/100))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_volatility/100))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_volatility/100))
        close_price = price
        
        # Generate volume
        volume = np.random.uniform(1000000, 5000000)
        
        data_rows.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
        price_values.append(close_price)
    
    df = pd.DataFrame(data_rows, index=dates)
    return df

def data_input_form():
    """Form for inputting stock data"""
    with st.form('Data Input'):
        st.write('Choose your data source:')
        
        data_source = st.radio(
            'Data Source',
            ['Yahoo Finance', 'Upload CSV', 'Sample Data'],
            help='Fetch live data, upload your own CSV, or use sample data for testing'
        )
        
        if data_source == 'Yahoo Finance':
            col1, col2 = st.columns([1, 1])
            with col1:
                ticker = st.text_input('Stock Ticker', value='AAPL', placeholder='e.g., AAPL, MSFT, GOOGL')
                st.caption("💡 Try: AAPL, MSFT, GOOGL, AMZN, NVDA")
            with col2:
                days_back = st.selectbox('Time Period', [7, 14, 30, 60, 90, 180, 365], index=2)
            
            start_date = datetime.now() - timedelta(days=days_back)
            end_date = datetime.now()
            
        elif data_source == 'Upload CSV':
            st.info("📁 **CSV Upload - Coming Soon!**")
            st.write("This feature will allow you to upload your own stock data CSV files.")
            st.write("For now, please use Yahoo Finance or Sample Data options.")
            
            # Disable the upload functionality
            uploaded_file = None
            ticker = st.text_input('Stock Name (for file naming)', value='CUSTOM', placeholder='e.g., MY_STOCK', disabled=True)
            
        else:  # Sample Data
            days_back = st.selectbox('Sample Data Period', [7, 14, 30, 60, 90], index=2)
            ticker = st.text_input('Sample Stock Name', value='SAMPLE', placeholder='e.g., TEST_STOCK')
        
        submit_data = st.form_submit_button('Load Data')
        
        if submit_data:
            if data_source == 'Yahoo Finance':
                if not ticker:
                    st.error('Please enter a stock ticker')
                    return
                
                with st.spinner("Fetching stock data..."):
                    stock_data = get_stock_data(ticker, start_date, end_date)
                    if stock_data is not None:
                        state['stock_data'] = stock_data
                        state['ticker'] = ticker
                        st.success(f"✅ Successfully loaded {len(stock_data)} days of data for {ticker}")
            
            elif data_source == 'Upload CSV':
                st.warning("📁 CSV upload is not yet available. Please use Yahoo Finance or Sample Data.")
                return
            
            else:  # Sample Data
                with st.spinner("Generating sample data..."):
                    stock_data = generate_sample_data(days_back)
                    state['stock_data'] = stock_data
                    state['ticker'] = ticker
                    st.success(f"✅ Generated {len(stock_data)} days of sample data for {ticker}")
                    st.info("🎵 This is sample data for testing. Try different musical presets!")

def data_visualization():
    """Display stock data visualization and summary"""
    stock_data = state['stock_data']
    ticker = state.get('ticker', 'Unknown')

    # --- Always normalize DataFrame columns for yfinance/CSV quirks ---
    stock_data = parse_yf_df(stock_data)
    
    # Show data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days", len(stock_data))
    with col2:
        try:
            avg_close = stock_data['Close'].mean()
            # Handle numpy/pandas types properly
            if hasattr(avg_close, 'item'):
                avg_close = float(avg_close.item())
            else:
                avg_close = float(avg_close)
            st.metric("Avg Close", f"${avg_close:.2f}")
        except:
            st.metric("Avg Close", "N/A")
    with col3:
        try:
            max_close = stock_data['Close'].max()
            if hasattr(max_close, 'item'):
                max_close = float(max_close.item())
            else:
                max_close = float(max_close)
            st.metric("Max Close", f"${max_close:.2f}")
        except:
            st.metric("Max Close", "N/A")
    with col4:
        try:
            min_close = stock_data['Close'].min()
            if hasattr(min_close, 'item'):
                min_close = float(min_close.item())
            else:
                min_close = float(min_close)
            st.metric("Min Close", f"${min_close:.2f}")
        except:
            st.metric("Min Close", "N/A")
    
    st.markdown("#### Stock Data Visualization")
    st.plotly_chart(make_linechart(stock_data), use_container_width=True)
    st.plotly_chart(make_barchart(stock_data), use_container_width=True)
    
    # Show data table (not inside any expander)
    st.markdown("#### Raw Data Table")
    st.dataframe(stock_data.tail(10))

def music_generation_form():
    """Form for generating music from stock data with multiple methods"""
    
    # Create tabs for different music generation methods
    tab1, tab2 = st.tabs(["🎵 Price to Pitch", "🎼 Volatile Progression"])
    
    with tab1:
        price_to_pitch_form()
    
    with tab2:
        volatile_progression_form()

def price_to_pitch_form():
    """Original price-to-pitch music generation method"""
    preset_name = st.selectbox(
        'Quick Start Preset',
        options=["Custom"] + list(MUSICAL_PRESETS.keys()),
        index=0,
        help="Choose a preset for quick setup"
    )
    
    # Show preset info if selected
    if preset_name != "Custom":
        preset = MUSICAL_PRESETS[preset_name]
        st.info(f"🎵 {preset_name} preset: {preset['description']}")
        st.success(f"**Scale:** {preset['scale']} | **BPM:** {preset['bpm']} | ")
    
    # Advanced settings with the form inside
    with st.form('Price to Pitch Generation'):
        col1, col2 = st.columns([1, 1])
        with col1:
            scale_name = st.selectbox(
                'Musical Scale',
                options=list(MUSICAL_SCALES.keys()),
                index=2,  # Default to C Minor Pentatonic
                help="Choose a musical scale for the composition"
            )
            bpm = st.slider(
                'BPM (Beats Per Minute)',
                min_value=60,
                max_value=200,
                value=120,
                help="Tempo of the generated music"
            )
        with col2:
            note_duration = st.selectbox(
                'Note Duration',
                options=[0.25, 0.5, 1.0, 2.0],
                index=0,
                format_func=lambda x: f"{x} beats",
                help="Duration of each note in beats"
            )
        
        # Data mapping
        st.write('🎯 Data Mapping:')
        pitch_source = st.selectbox(
            'Pitch Source',
            options=["Close", "High", "Low", "Open"],
            index=0,
            help="Which price data to use for note pitch"
        )
        
        # --- Submit button ---
        generate_music = st.form_submit_button('🎵 Generate Price-to-Pitch Music')
        
        if generate_music:
            generate_price_to_pitch_music(preset_name, scale_name, bpm, note_duration, pitch_source)

def volatile_progression_form():
    """Volatile progression music generation method using scale.pattern and arpeggio"""
    st.markdown("### 🎼 Volatile Progression Method (Advanced)")
    st.write("Generate chord progressions based on stock volatility and volume patterns, using musicpy's scale.pattern and arpeggio.")
    
    with st.form('Volatile Progression Generation'):
        bpm = st.slider(
                'BPM (Beats Per Minute)',
                min_value=60,
                max_value=200,
                value=120,
                help="Tempo of the generated music"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            scale_name = st.selectbox(
                'Scale',
                options=["C major", "C minor", "D major", "E minor", "G major", "A minor"],
                index=0,
                help="Choose the musical scale"
            )
            progression_str = st.text_input(
                'Chord Progression (scale degrees)',
                value='1465',
                help="Enter progression as scale degrees, e.g., 1465 for I-IV-vi-V"
            )
            chord_size = st.selectbox(
                'Chord Size (num notes)',
                options=[3, 4, 5],
                index=1,
                help="Number of notes per chord (triad, 7th, 9th, etc.)"
            )
        with col2:
            chord_duration = st.selectbox(
                'Chord Duration',
                options=[1, 2, 4],
                index=1,
                format_func=lambda x: f"{x} bars",
                help="Duration of each chord in bars"
            )

        
        # --- Visualize volume and volatility ---
        if state.get('stock_data') is not None:
            n_chords = len(progression_str)
            stock_data = state['stock_data']
            volume_series = stock_data['Volume']
            price_series = stock_data['Close']
            chunked_volume_series, chunked_volatility_series = calculate_chunk_metrics(stock_data, n_chords)
            chunk_size = len(stock_data) // n_chords
            chunk_x = [stock_data.index[i*chunk_size] for i in range(n_chords)]
            # Calculate rolling volatility for visualization
            rolling_vol = price_series.pct_change().rolling(window=chunk_size, min_periods=1).std()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=volume_series, name='Volume', yaxis='y1', ))
            fig.add_trace(go.Scatter(x=chunk_x, y=chunked_volume_series, name='Chunked Volume', yaxis='y1', mode='markers+lines', ))
            fig.add_trace(go.Scatter(x=stock_data.index, y=rolling_vol, name='Rolling Volatility', yaxis='y2', ))
            fig.add_trace(go.Scatter(x=chunk_x, y=chunked_volatility_series, name='Chunked Volatility', yaxis='y2', mode='markers+lines', ))
            fig.update_layout(
                title='Volume and Volatility (Chunked and Original)',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Volume', side='left', showgrid=False),
                yaxis2=dict(title='Volatility', overlaying='y', side='right', showgrid=False),
                legend=dict(orientation='h'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        # --- Submit button ---
        generate_music = st.form_submit_button('🎼 Generate Volatile Progression')
        
        if generate_music:
            generate_volatile_progression_music_pattern(
                scale_name, progression_str, chord_size, chord_duration, bpm
            )

def results_section():
    """Display results and download options"""
    if 'composition_details' in state:
        details = state['composition_details']
        
        # Show composition details
        st.write("### 🎼 Composition Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Scale", str(details.get('scale', 'Unknown')))
            st.metric("BPM", int(details.get('bpm', 0)))
        with col2:
            duration = details.get('duration', 0)
            st.metric("Duration", f"{duration:.1f}s" if isinstance(duration, (int, float)) else "N/A")
            st.metric("Total Notes", int(details.get('total_notes', 0)))
        with col3:
            st.metric("Pitch Source", str(details.get('pitch_source', 'Unknown')))

        # Plot chords
        fig = plot_chords(state['stock_song'], start_time=0, end_time=48)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        if 'midi_bytes' in state and state['midi_bytes'] is not None:
            download_midi_no_refresh(state['midi_filename'], state['midi_bytes'])
        
        # Play button (placeholder for future audio playback)
        play_button = st.button('🎵 Play Preview')
        if play_button:
            if 'stock_song' in state and state['stock_song'] is not None:
                with st.spinner("Playing audio..."):
                    try:
                        # Get BPM from composition details
                        bpm = state.get('composition_details', {}).get('bpm', 120)
                        audio_obj = play_audio(state['stock_song'], bpm=bpm)
                        
                        if audio_obj is not None:
                            # If pygame is not available, show audio player
                            st.audio(audio_obj, format='audio/wav')
                        else:
                            st.success("🎵 Audio playing! (If you don't hear anything, check your system audio)")
                    except Exception as e:
                        st.error(f"Error playing audio: {str(e)}")
                        st.info("💡 Try downloading the MIDI file and importing it into your DAW (FL Studio, Ableton, etc.)")
            else:
                st.error("No music generated yet. Please generate music first.")

def confused():
    st.warning("Don't know what to do?")
    st.markdown(help_md)

@st.experimental_fragment
def download_midi_no_refresh(midi_fname, midi_bytes):
    """ 
    st.download_button() refreshes the page. The workaround is to wrap it in a fragment. 
    It is possible for the first download to fail.
    """
    st.download_button(
        label='💾 Download MIDI File', 
        data=midi_bytes, 
        file_name=midi_fname, 
        mime='audio/midi',
        use_container_width=True
    )

help_md = """
## 🎵 How Money to Music Works

**The Four-Step Workflow:**

1. **📈 Get Stock Data**
   - Fetch from Yahoo Finance OR upload your own CSV
   - Supports OHLCV (Open, High, Low, Close, Volume) data

2. **📊 Review Data**
   - Visualize stock price and volume charts
   - Check data quality and time period

3. **🎵 Generate Music**
   - Choose from presets or customize settings
   - Map prices to pitch, volume to velocity
   - Apply musical scales for harmony

4. **🎼 Download & Play**
   - Download MIDI file for your DAW
   - Import to FL Studio, Ableton, etc.
   - Add effects and arrangement

## 🎯 Data Mapping

**Price → Pitch:**
- Higher prices = higher notes
- Lower prices = lower notes
- Scaled to musical scale

## 🎼 Musical Presets

- **Ambient:** Chill, atmospheric vibes with long, sustained notes
- **Electronic:** High-energy electronic with sharp, rhythmic patterns  
- **Jazz:** Sophisticated jazz with swing and improvisation feel
- **Rock:** High-energy rock with driving rhythms
- **Blues:** Soulful blues with expressive bends and slides

## 💡 Tips

- **Volatile stocks** create more dynamic melodies
- **Longer time periods** create longer compositions
- **High-volume stocks** provide better velocity variation
- Try different presets for different moods
- Import to FL Studio for full arrangement
"""

def generate_price_to_pitch_music(preset_name, scale_name, bpm, note_duration, pitch_source):
    """Generate music using the original price-to-pitch method"""
    # Get parameters (from preset or manual)
    if preset_name != "Custom":
        preset = MUSICAL_PRESETS[preset_name]
        scale_name = preset["scale"]
        bpm = preset["bpm"]
        note_duration = preset["note_duration"]
        pitch_source = preset["pitch_source"]
    
    # Ensure we have valid data
    if state['stock_data'] is None:
        st.error("No stock data available. Please load data first.")
        return
    
    # Generate MIDI
    with st.spinner("Creating MIDI file..."):
        midi_file = create_midi_from_stock_data(
            state['stock_data'], scale_name, bpm, note_duration, pitch_source
        )
        
        if midi_file:
            # Save MIDI to bytes using the robust utility
            midi_bytes = export_to_midi_as_bytes(midi_file)
            
            # Store in session state
            state['stock_song'] = midi_file
            state['midi_bytes'] = midi_bytes
            ticker_name = state.get('ticker', 'stock')
            if ticker_name is None:
                ticker_name = 'stock'
            state['midi_filename'] = f"{ticker_name}_{scale_name.replace(' ', '_')}_{bpm}bpm.mid"
            state['composition_details'] = {
                'scale': scale_name,
                'bpm': bpm,
                'note_duration': note_duration,
                'pitch_source': pitch_source,
                'total_notes': len(state['stock_data']),
                'duration': len(state['stock_data']) * note_duration / bpm * 60
            }
            
            st.success("🎵 MIDI file generated successfully!")

def custom_arpeggiate_chord(chord, interval, total_duration):
    notes = chord.notes  # Use the full note objects, not just names
    n_notes = len(notes)
    
    # Calculate how many complete up-down cycles we can fit
    notes_per_cycle = 2 * n_notes - 2 if n_notes > 1 else 1
    n_cycles = int(total_duration // (interval * notes_per_cycle))
    leftover = total_duration - n_cycles * interval * notes_per_cycle
    
    # Create up-down pattern: 1,2,3,4,3,2,1,2,3,4,3,2,1...
    sequence = []
    for cycle in range(n_cycles):
        # Up: 0,1,2,3,4...
        for i in range(n_notes):
            sequence.append(notes[i])
        # Down: n-2, ..., 1 (skip the top note to avoid repetition)
        for i in range(n_notes - 2, 0, -1):
            sequence.append(notes[i])
    
    # Handle leftover time
    if leftover > 0:
        n_left = int(leftover // interval)
        if n_left > 0:
            # Add partial up-down pattern for leftover
            for i in range(min(n_left, n_notes)):
                sequence.append(notes[i])
            if n_left > n_notes:
                for i in range(min(n_left - n_notes, n_notes - 1), 0, -1):
                    sequence.append(notes[i])
    
    durations = [interval] * len(sequence)
    intervals = [interval] * len(sequence)
    return mp.chord(sequence) % (durations, intervals)

def generate_volatile_progression_music_pattern(scale_name, progression_str, chord_size, chord_duration, bpm):
    """Generate music using scale.pattern, volume binning for octave, volatility for arpeggio interval"""
    if state['stock_data'] is None:
        st.error("No stock data available. Please load data first.")
        return
    
    with st.spinner("Creating volatile progression with arpeggios..."):
        # Step 1: Get finance series
        n_chords = len(progression_str)
        finance_series = state['stock_data']['Close']
        chunked_series = chunk_finance_series(finance_series, n_chords)
        chunked_volume_series, chunked_volatility_series = calculate_chunk_metrics(state['stock_data'], n_chords)
        
        # Step 2: Get chords from scale pattern
        scale_obj = mp.S(scale_name)
        chords = scale_obj.pattern(progression_str, num=chord_size, duration=chord_duration, interval=0)
        # chords is a list of musicpy chord objects
        
        # Step 3: Volume binning for octave
        octaves = get_octave_bins(chunked_volume_series)
        # Step 4: Volatility quantile for arpeggio interval
        intervals = get_arpeggio_intervals(chunked_volatility_series)
        
        # Step 5: For each chunk, transpose chord to octave, arpeggiate with interval
        arps = []
        for i, chord in enumerate(chords):
            octave = octaves[i % len(octaves)]
            interval = intervals[i % len(intervals)]
            
            # Get the current root note's MIDI pitch number
            current_root_midi = chord[0].degree
            
            # Get the current root note's name
            current_root_note = chord[0].name
            
            # Calculate what the MIDI number should be for the target octave
            # Use musicpy's note_to_degree to get the target MIDI number
            target_root_midi = mp.note_to_degree(f"{current_root_note}{octave}")
            
            # Calculate semitones to transpose
            semitones = target_root_midi - current_root_midi
            chord_transposed = chord + semitones
            arp = custom_arpeggiate_chord(chord_transposed, interval, chord_duration)
            arps.append(arp)
            print(arp)
        
        # Step 6: Combine all arpeggios into a piece
        if arps:
            final_piece = arps[0]
            for arp in arps[1:]:
                final_piece = final_piece | arp
            final_piece.bpm = bpm
            midi_bytes = export_to_midi_as_bytes(final_piece)
            state['stock_song'] = final_piece
            state['midi_bytes'] = midi_bytes
            ticker_name = state.get('ticker', 'stock')
            if ticker_name is None:
                ticker_name = 'stock'
            state['midi_filename'] = f"{ticker_name}_progression_{bpm}bpm.mid"
            state['composition_details'] = {
                'method': 'Volatile Progression',
                'progression': progression_str,
                'scale': scale_name,
                'bpm': bpm,
                'chord_duration': chord_duration,
                'chord_size': chord_size,
                'total_chunks': len(chunked_series),
                'duration': len(chunked_series) * chord_duration * 4 / bpm * 60  # 4 beats per bar
            }
            st.success("🎼 Volatile progression generated successfully!")

def get_octave_bins(volume_series):
    """Map volume percentiles to octaves (C3 to C6)"""
    percentiles = np.percentile(volume_series, [10,20,30,40,50,60,70,80,90])
    octaves = []
    for v in volume_series:
        if v <= percentiles[0]:
            octaves.append(3)
        elif v <= percentiles[1]:
            octaves.append(3)
        elif v <= percentiles[2]:
            octaves.append(4)
        elif v <= percentiles[3]:
            octaves.append(4)
        elif v <= percentiles[4]:
            octaves.append(4)
        elif v <= percentiles[5]:
            octaves.append(5)
        elif v <= percentiles[6]:
            octaves.append(5)
        elif v <= percentiles[7]:
            octaves.append(5)
        else:
            octaves.append(6)
    return octaves

def get_arpeggio_intervals(volatility_series):
    """Map volatility quantiles to arpeggio intervals"""
    quantiles = np.quantile(volatility_series, [0.25, 0.5, 0.75])
    intervals = []
    for v in volatility_series:
        if v <= quantiles[0]:
            intervals.append(1.0)  # whole note
        elif v <= quantiles[1]:
            intervals.append(0.5)  # half note
        elif v <= quantiles[2]:
            intervals.append(0.25)  # quarter note
        else:
            intervals.append(0.125)  # eighth note
    return intervals

def chunk_finance_series(series, num_chunks):
    """Chunk the finance series into periods"""
    chunk_size = len(series) // num_chunks
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(series)
        chunk = series.iloc[start_idx:end_idx]
        chunks.append(chunk)
    
    return chunks

def calculate_chunk_metrics(stock_data, num_chunks):
    """Calculate volume and volatility metrics for each chunk"""
    volume_series = stock_data['Volume']
    price_series = stock_data['Close']
    
    chunk_size = len(stock_data) // num_chunks
    volume_metrics = []
    volatility_metrics = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(stock_data)
        
        # Calculate average volume for chunk
        chunk_volume = volume_series.iloc[start_idx:end_idx].mean()
        volume_metrics.append(chunk_volume)
        
        # Calculate volatility (standard deviation of price changes)
        chunk_prices = price_series.iloc[start_idx:end_idx]
        price_changes = chunk_prices.pct_change().dropna()
        volatility = price_changes.std() if len(price_changes) > 1 else 0
        volatility_metrics.append(volatility)
    
    return volume_metrics, volatility_metrics
    