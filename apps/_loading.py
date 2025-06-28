import time
import streamlit as st
from hydralit import HydraHeadApp
from hydralit_components import HyLoader, Loaders

class MyLoadingApp(HydraHeadApp):
  def __init__(self, title = 'Loader', delay=0,loader=Loaders.standard_loaders, **kwargs):
    self.__dict__.update(kwargs)
    self.title = title
    self.delay = delay
    self._loader = loader

  def run(self,app_target):
    try:
      loader_txt = """
      <style> 
#rcorners1 {
border-radius: 0px;
background: grey;
color: #00000;
alignment: center;
opacity: 0.90;
padding: 5px; 
width: 920px;
height: 80px; 
z-index: 9998; 
}
#banner {
color: white;
vertical-align: text-top;
text-align: center;
z-index: 9999; 
}
</style>
<div id="rcorners1">
<h2 id="banner">💰 Money to Music 🎵 - Loading...</h2>
<br>
</div>
      """
      app_title = ''
      if hasattr(app_target,'title'):
        app_title = app_target.title

      if app_title == 'Loader Playground': # debug purpose
        app_target.run()
      else:
        with HyLoader('Preparing your stock-to-music converter...', Loaders.pretty_loaders, index=0):
          app_target.run()
  
    except Exception as e:
      # st.image("./resources/failure.png",width=100,)
      # st.error('An error has occurred, that shouldn\'t happen...')
      st.error('Error details: {}'.format(e))
      st.exception(e)