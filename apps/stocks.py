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
    prices = []
    volumes = []
    
    for i in range(days):
        # Random walk for price
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 2)  # Daily change with std dev of 2
            price = prices[-1] * (1 + change/100)
        
        # Generate OHLC from base price
        daily_volatility = np.random.uniform(0.5, 2.0)
        open_price = price * (1 + np.random.uniform(-daily_volatility/100, daily_volatility/100))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_volatility/100))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_volatility/100))
        close_price = price
        
        # Generate volume
        volume = np.random.uniform(1000000, 5000000)
        
        prices.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
        volumes.append(volume)
    
    df = pd.DataFrame(prices, index=dates)
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
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="CSV should have columns: Date, Open, High, Low, Close, Volume"
            )
            ticker = st.text_input('Stock Name (for file naming)', value='CUSTOM', placeholder='e.g., MY_STOCK')
            
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
                if uploaded_file is None:
                    st.error('Please upload a CSV file')
                    return
                
                with st.spinner("Parsing CSV data..."):
                    stock_data = parse_csv_data(uploaded_file)
                    if stock_data is not None:
                        state['stock_data'] = stock_data
                        state['ticker'] = ticker
                        st.success(f"✅ Successfully loaded {len(stock_data)} days of data from CSV")
            
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
            if hasattr(avg_close, 'item'):
                avg_close = avg_close.item()
            st.metric("Avg Close", f"${avg_close:.2f}")
        except:
            st.metric("Avg Close", "N/A")
    with col3:
        try:
            max_close = stock_data['Close'].max()
            if hasattr(max_close, 'item'):
                max_close = max_close.item()
            st.metric("Max Close", f"${max_close:.2f}")
        except:
            st.metric("Max Close", "N/A")
    with col4:
        try:
            min_close = stock_data['Close'].min()
            if hasattr(min_close, 'item'):
                min_close = min_close.item()
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
    """Form for generating music from stock data"""
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
        # st.write(f"**Scale:** {preset['scale']} | **BPM:** {preset['bpm']} | **Melody:** {preset['melody_instrument']}")
    
    # Advanced settings with the form inside
    with st.form('Music Generation'):
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
        generate_music = st.form_submit_button('🎵 Generate Music')
        
        if generate_music:
            # Get parameters (from preset or manual)
            if preset_name != "Custom":
                preset = MUSICAL_PRESETS[preset_name]
                scale_name = preset["scale"]
                bpm = preset["bpm"]
                note_duration = preset["note_duration"]
                pitch_source = preset["pitch_source"]
            
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
                    state['midi_filename'] = f"{state.get('ticker', 'stock')}_{scale_name.replace(' ', '_')}_{bpm}bpm.mid"
                    state['composition_details'] = {
                        'scale': scale_name,
                        'bpm': bpm,
                        'note_duration': note_duration,
                        'pitch_source': pitch_source,
                        'total_notes': len(state['stock_data']),
                        'duration': len(state['stock_data']) * note_duration / bpm * 60
                    }
                    
                    st.success("🎵 MIDI file generated successfully!")

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
    