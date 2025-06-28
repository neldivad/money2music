from streamlit import session_state as state

def check_pygame_compatibility():
    if 'pygame_compatible' not in state:
        import pygame
        try:
            pygame.mixer.init(44100, -16, 2, 1024)
            state['pygame_compatible'] = True
            return True
        
        except Exception:
            state['pygame_compatible'] = False
            return False
        
    else:
        return state['pygame_compatible']