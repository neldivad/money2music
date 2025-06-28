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
   streamlit run main.py
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

## Deployment
You need `packages.txt` to install fluidsynth for stcloud to wokr. 

---

