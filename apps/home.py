import streamlit as st 
import streamlit.components.v1 as components


def main():
    # Hero section with the new messaging
    st.markdown("""
    # 🎵 Money2Music
    ### Making music with money
    
    **Watching your portfolio is not the peak. Hear how your portfolio is doing.**
    
    Transform financial data into beautiful music. Turn price movements into melodies, 
    volatility into rhythms, and market trends into harmonies.
    """)
    
    # Simple call-to-action
    st.markdown("""
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Getting started guide
    st.markdown("## 📋 How to Get Started")
    
    with st.expander("🎯 Quick Start Guide", expanded=True):
        st.markdown("""
        ### 6 Simple Steps:
        1. **Go to Stocks** - Navigate to the Stocks section above
        2. **Select Ticker** - Choose any stock symbol (e.g., AAPL, TSLA, BTC-USD)
        3. **Load Data** - Fetch the latest price data
        4. **Select Scale** - Choose your preferred musical scale or preset
        5. **Generate** - Create your musical composition
        6. **Play** - Listen to your portfolio's melody!
        """)
        
        st.info("💡 **Pro Tip:** Try different time periods and scales to create unique compositions!")
    
    st.divider()
    
    # Author notes section
    st.markdown("## 🎼 About This Project")
    
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### How It Works
        This is a fun proof of concept that demonstrates the relationship between financial data and music:
        
        - **Simple Mapping**: Noisy price data is normalized to harmonic scales
        - **Real-time Conversion**: Stock movements become musical notes
        - **Instant Composition**: Generate unique melodies from any ticker
        
        ### Future Possibilities
        For more complex implementations, we could explore:
        - **Chord Progressions**: Set specific chord sequences
        - **Volume as Arpeggiation Range**: Use trading volume to control note range
        - **Price Volatility as Arpeggiation Density**: Higher volatility = more complex patterns
        - **Multi-instrument Arrangements**: Different instruments for different data points
        - **Tempo Mapping**: Use market speed to control musical tempo
        """)
    
    st.markdown("""
    ---
    **🎯 Project Status:** Just for fun and proof of concept. If this gets traction I'll work harder.
    
    **💡 Want to contribute?** Give suggestions in the Discord server!
    """)