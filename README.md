# 💰 Money to Music 🎵

Transform stock market data into musical compositions! This Streamlit app converts historical stock data (OHLCV) into MIDI files that you can import into your favorite DAW (like FL Studio) to create unique musical pieces.

## 🚀 Features

- **📈 Real-time Stock Data**: Fetch live stock data using Yahoo Finance
- **🎼 Multiple Musical Scales**: Choose from 8 different musical scales (Major, Minor, Pentatonic, Blues, etc.)
- **🎛️ Customizable Parameters**: Adjust BPM, note duration, and data mapping
- **📊 Interactive Visualization**: View stock price and volume charts
- **💾 MIDI Export**: Download generated MIDI files for use in your DAW
- **🎯 Smart Data Mapping**: Map prices to pitch and volume to velocity

## 🎵 How It Works

### The Three-Step Workflow

1. **📈 Acquire the Data**: Get raw numbers from a stock (OHLCV)
2. **🎼 Translate the Data**: Convert numbers into musical format (MIDI)
3. **🎵 Produce the Sound**: Import MIDI into FL Studio and create your vibe

### Data Mapping

- **Price → Pitch**: Higher prices = higher notes, scaled to musical scale
- **Volume → Velocity**: Higher volume = louder notes for dynamic expression
- **Time → Rhythm**: Each trading day becomes a musical note

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Run the app**:
   ```bash
   streamlit run apps/home.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Configure your composition**:
   - Enter a stock ticker (e.g., TSLA, AAPL, MSFT)
   - Select date range
   - Choose musical scale
   - Adjust BPM and note duration
   - Click "Generate Music"

4. **Download and use**:
   - Download the generated MIDI file
   - Import into FL Studio, Ableton, or your preferred DAW
   - Add effects, instruments, and arrangement

## 🎼 Musical Scales

- **C Minor Pentatonic**: Vibey, hard to sound bad (recommended for beginners)
- **C Major**: Bright, uplifting
- **C Minor**: Dark, emotional
- **C Blues**: Soulful, expressive
- **C Dorian**: Jazz-like, sophisticated
- **C Mixolydian**: Rock, blues feel
- **C Lydian**: Dreamy, ethereal

## 🎯 Tips for Better Results

### Stock Selection
- **Volatile stocks** create more dynamic melodies
- **Longer time periods** create longer compositions
- **High-volume stocks** provide better velocity variation

### Musical Parameters
- **Lower BPM (60-90)**: More ambient, chill vibes
- **Higher BPM (120-160)**: More energetic, danceable
- **Shorter note durations**: More notes, busier composition
- **Longer note durations**: Fewer notes, more spacious

### DAW Workflow
1. Import the MIDI file
2. Add a bass instrument for the melody
3. Add drums and percussion
4. Layer with pads or strings
5. Add effects (reverb, delay, compression)
6. Arrange and mix

## 📊 Example Workflow

1. **Select TSLA** for the last 30 days
2. **Choose C Minor Pentatonic** scale
3. **Set BPM to 120** and note duration to 0.25 beats
4. **Generate** and download the MIDI
5. **Import to FL Studio**:
   - Add a synth bass for the melody
   - Add a drum machine
   - Add reverb and delay effects
   - Arrange into a full song

## 🔧 Technical Details

### Dependencies
- `streamlit`: Web app framework
- `yfinance`: Stock data fetching
- `pandas`: Data manipulation
- `mido`: MIDI file creation
- `plotly`: Interactive charts
- `numpy`: Numerical operations

### Data Sources
- **Yahoo Finance**: Real-time and historical stock data
- **OHLCV Data**: Open, High, Low, Close, Volume

### MIDI Format
- **Standard MIDI file** (.mid)
- **Compatible** with all major DAWs
- **Single track** with melody notes
- **Configurable tempo** and note duration

## 🎵 Creative Possibilities

- **Market Analysis**: Hear market trends through music
- **Algorithmic Composition**: Create unique musical patterns
- **Data Sonification**: Make financial data accessible through sound
- **Educational Tool**: Learn about markets and music simultaneously
- **Artistic Expression**: Create music that reflects market movements

## 🤝 Contributing

Feel free to contribute by:
- Adding new musical scales
- Improving the data mapping algorithms
- Adding more visualization options
- Creating preset configurations
- Adding support for other data sources

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

---

**Happy composing! 🎵💰** 