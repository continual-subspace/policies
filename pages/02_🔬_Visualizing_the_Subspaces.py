import streamlit as st
import os
import pickle
from utils import *
inject_custom_css('assets/styles.css')

scenarios = {
    "Forgetting scenario":["hugefeet","moon","carrystuff","rainfall"],
    "Transfer scenario":["carrystuff_hugegravity","moon","defective_modules","hugefeet_rainfall"],
    "Distraction scenario":["normal","inverted_actions","normal","inverted_actions"],
    "Composability scenario":["tinyfeet","moon","carrystuff_hugegravity","tinyfeet_moon"]
}

#st.title("Visualizing the subspaces"
st.markdown("<h3 style='text-align: left';>Visualizing the reward landscapes of the Subspaces</h3>",unsafe_allow_html = True)
with st.expander("Reward Landscape"):
    row_0_1,row_0_2 = st.columns([1,3])
with row_0_1:
    benchmark = st.selectbox('',["Forgetting scenario","Transfer scenario","Distraction scenario","Composability scenario"],index=0)
    path = "../data/halfcheetah_benchmark"+str(0)+"/"
    n_anchors = st.slider('number of anchors:', 3, 3, 4)
    task = st.selectbox('Task',scenarios[benchmark],index=0)
    with open(path+str(5)+"/eval.pkl", "rb") as f:
        data = pickle.load(f)["stage_"+str(n_anchors)]["task_"+str(task)]
    alphas, rewards, q1 = data["alphas"],data["rewards"],data["values1"]
    fig = display_kshot(alphas,rewards,"reward")
    if n_anchors == 4:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig)
    #alphas, rewards, q1 = data["alphas"],data["rewards"],data["values1"]
    #cov = ((rewards - rewards.mean()) * (q1 - q1.mean())).sum() / (rewards.shape[0] - 1)
    #std1 = rewards.std()
    #std2 = q1.std()
    #corr = round((cov / (std1 * std2)).item(),2)
#row_2_1,row_2_2,row_2_3  = st.columns([1,2,2])
#with row_2_1:
#    benchmark = st.selectbox('Scenario:',["1. Negative backward","3. Distraction","4. Compositionality"],index=2)
#    path = os.getcwd()+"/data/halfcheetah_benchmark"+benchmark[0]+"/"
#    seed = st.selectbox('Seed:',[int(d)for d in os.listdir(path) if os.path.isfile(path+d+"/eval.pkl")],index=0)
#    stage = st.slider('Stage:', 1, 3, 3) 
#    task = st.slider('Task:', 0, 3, 3)
#    with open(path+str(seed)+"/eval.pkl", "rb") as f:
#        data = pickle.load(f)["stage_"+str(stage)]["task_"+str(task)]
#    alphas, rewards, q1 = data["alphas"],data["rewards"],data["values1"]
#    cov = ((rewards - rewards.mean()) * (q1 - q1.mean())).sum() / (rewards.shape[0] - 1)
#    std1 = rewards.std()
#    std2 = q1.std()
#    corr = round((cov / (std1 * std2)).item(),2)
#    
#with row_2_2:
#    st.markdown("<h6 style='text-align: center; color: black;'>Reward</h6>", unsafe_allow_html=True)
#    fig = display_kshot(alphas,rewards,"reward")
#    if stage == 3:
#        st.plotly_chart(fig, use_container_width=True)
#    else:
#        st.pyplot(fig)
#
#with row_2_3:
#    st.markdown("<h6 style='text-align: center; color: black;'>Q estimation</h6>", unsafe_allow_html=True)
#    fig = display_kshot(alphas,q1,"q")
#    if stage == 3:
#        st.plotly_chart(fig, use_container_width=True)
#    else:
#        st.pyplot(fig)
#_,row_3_2 = st.columns([1,4])
#with row_3_2:
#    st.markdown("<h4 style='text-align: center; color: black;'>Correlation = "+str(corr)+"</h4>", unsafe_allow_html=True)