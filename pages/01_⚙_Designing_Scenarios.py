
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import codecs
import pandas as pd
from utils import *
from PIL import Image

def run():
    inject_custom_css('assets/styles.css')
    d = {
        "Forgetting scenario":"<b style='color: #ff4b4b'>Forgetting scenarios</b> are designed such that a single policy tends to forget the former task when learning a new one.",
        "Transfer scenario":"<b style='color: #ff4b4b'>Transfer scenarios</b> are designed such that a single policy has more difficulties to learn a new task after having learned the former one, rather than learning it from scratch. ",
        "Distraction scenario":"<b style='color: #ff4b4b'>Distraction scenarios</b> alternate between a normal task and a very different distraction task that disturbs the whole learning process of a single policy",
        "Composability scenario":"<b style='color: #ff4b4b'>Composability scenarios</b> present two first tasks that will be useful to learn the last one, but a very different distraction task is put at the third place to disturb this forward transfer.",
    }

    scenarios = {
        "Forgetting scenario":["hugefeet","moon","carrystuff","rainfall"],
        "Transfer scenario":["carrystuff_hugegravity","moon","defective_modules","hugefeet_rainfall"],
        "Distraction scenario":["normal","inverted_actions","normal","inverted_actions"],
        "Composability scenario":["tinyfeet","moon","carrystuff_hugegravity","tinyfeet_moon"]
    }

    data_perf = {"method":["FT-1","FT-L2","EWC","PNN","SAC-N","FT-N","<b style='color: #ff4b4b'>CSP</b>"]*4,
            "performance":[0.95,1.07,1.37,1.35,1.0,1.68,1.69] + \
                        [0.42,0.37,0.52,0.87,1.0,0.89,0.97] + \
                        [0.36,0.61,0.66,0.97,1.0,0.84,1.27] + \
                        [1.28,1.19,1.38,1.04,1.0,1.48,1.42],
            "scenario":["Forgetting scenario"]*7+["Transfer scenario"]*7+["Distraction scenario"]*7+["Composability scenario"]*7
        }
    data_perf = pd.DataFrame(data_perf)

    #st.title('Designing the scenarios with Brax')
    st.markdown("<h3 style='text-align: left';>Designing Scenarios for Continual Learning</h3>",unsafe_allow_html = True)
    row_1_1,row_1_2  = st.columns([2,2])
    with row_1_1:
        st.markdown('''
            <div style='text-align: justify; padding: 2em 2em 2em 0';>
            Using Brax physics engine, we performed an in-depth study of our method by designing a number of continuous control scenarios to
            separately evaluate capabilities specific to CRL agents. For
            each capability, we create two scenarios: one based on HalfCheetah and one based on Ant.
            To do so, we generated a large number of environments by changing the dynamics of the standard environment. We
            aimed to ground our environment variations in realistic situations such as increased or decreased
            gravity, friction, or limb lengths. Then, we design task sequences based on the transfer matrices for
            these environments. We tested both the short-term (4 tasks, each with a budget of 1M interactions) and long term (8 tasks : short-term repeated twice) 
            versions of these scenarios. We compared CSP to other baselines. Among them, FT-N aims to fine-tune and clone the model everytime a new task appears.
            This untractable method is yet outperformed by CSP in the majority of the scenarios we designed. A demo of the task sequences solved by an optimal policy
            is available for each Halfcheetah scenario.
            </div>''', unsafe_allow_html = True)


    with row_1_2:
        benchmark = st.selectbox('',["Forgetting scenario","Transfer scenario","Distraction scenario","Composability scenario"],index=0)
        layout = go.Layout( margin=go.layout.Margin(l=0, r=0, b=0, t=0))
        fig = go.Figure(layout=layout)
        data = data_perf[data_perf["scenario"] == benchmark]

        fig.add_trace(go.Bar(
                        #data,
                        x=data['method'],
                        y=data['performance'],
                        hovertemplate="<br>%{y} </br><extra></extra>",
                        marker_color=["#44aa99"]*6+["#ff4b4b"]
                        )
                    )
        font = dict(family="sans-serif",size=16)
        fig.update_layout(showlegend=False, font = font,yaxis_title="Performance", height=250)
        fig.update_yaxes(range=[0,2.])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='text-align: center'>"+d[benchmark]+"</div>", unsafe_allow_html = True)
    with st.expander(""+benchmark+" demo", expanded=True):
        row_2_1,row_2_2,row_2_3,row_2_4  = st.columns([1,1,1,1])
        with row_2_1:
            st.markdown("<h6 style='text-align: center; color: black; font-weight: 400;'>"+scenarios[benchmark][0]+"</h6>", unsafe_allow_html=True)
            file = codecs.open("data/trajectories/hc_"+scenarios[benchmark][0]+".html", "r", "utf-8").read()
            components.html(file)
        with row_2_2:
            st.markdown("<h6 style='text-align: center; color: black; font-weight: 400;'>"+scenarios[benchmark][1]+"</h6>", unsafe_allow_html=True)
            file = codecs.open("data/trajectories/hc_"+scenarios[benchmark][1]+".html", "r", "utf-8").read()
            components.html(file)
        with row_2_3:
            st.markdown("<h6 style='text-align: center; color: black; font-weight: 400;'>"+scenarios[benchmark][2]+"</h6>", unsafe_allow_html=True)
            file = codecs.open("data/trajectories/hc_"+scenarios[benchmark][2]+".html", "r", "utf-8").read()
            components.html(file)
        with row_2_4:
            st.markdown("<h6 style='text-align: center; color: black; font-weight: 400;'>"+scenarios[benchmark][3]+"</h6>", unsafe_allow_html=True)
            file = codecs.open("data/trajectories/hc_"+scenarios[benchmark][3]+".html", "r", "utf-8").read()
            components.html(file)

if __name__ == "__main__":
    img = Image.open("data/images/icon.png")
    st.set_page_config(
        page_title="Continual Subspace of Policies",
        page_icon=img,
        layout="wide",
    )
    st.sidebar.markdown("<b>About</b>", unsafe_allow_html = True)
    st.sidebar.markdown("<div>We designed specific scenarios for different continual learning challenges. Look at the demo to get some insights about them !</div>", unsafe_allow_html = True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    run()