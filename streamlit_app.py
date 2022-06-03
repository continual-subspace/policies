import streamlit as st
from utils import *
from views import method, scenario, visualizing
from PIL import Image

def run_UI():
    route = get_current_route()
    if route == "about":
        method.run()

    elif route == 'scenarios':
        scenario.run()

    elif route == 'subspace':
        visualizing.run()
        
if __name__ == "__main__":
    img = Image.open("data/images/icon.png")
    st.set_page_config(
        page_title="Continual Subspace of Policies",
        page_icon=img,
        layout="wide",
        menu_items={}
    )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    inject_custom_css('assets/styles.css')
    
    run_UI()
    navbar_component()
