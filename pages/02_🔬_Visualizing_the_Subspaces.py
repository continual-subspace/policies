import streamlit as st
import pickle
from utils import *
from PIL import Image

def run():
    scenarios = {
        "Forgetting_scenario":["hugefeet","moon","carrystuff","rainfall"],
        "Transfer_scenario":["carrystuff_hugegravity","moon","defective_modules","hugefeet_rainfall"],
        "Distraction_scenario":["normal","inverted_actions","normal","inverted_actions"],
        "Composability_scenario":["tinyfeet","moon","carrystuff_hugegravity","tinyfeet_moon"]
    }

    path = "data/landscapes/subspace_points.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    #st.title("Visualizing the subspaces"
    st.markdown("<h3 style='text-align: left';>Visualizing the reward landscapes of the Subspaces</h3>",unsafe_allow_html = True)
    row_0_1,row_0_2  = st.columns([5,2])
    with row_0_1:
        st.markdown('''
            <div style='text-align: justify; padding: 2em 2em 2em 0';>
            Analyzing the diversity of performance within the subspace helps to understand which role each anchor plays during training. To do so, we split the subspace in N=8192 evenly separated points,
            defining <b style='color: #ff4b4b'>8192 different policies</b> that we evaluated 5 times each to get their average performance. Both the <b style='color: #ff4b4b'>smoothness</b> and the <b style='color: #ff4b4b'>diversity</b>
            of the figures gave us the intuition that this singular structure can be leveraged during training and evaluation.
            </div>''', unsafe_allow_html = True)
        st.markdown('''
            <div style='text-align: justify; padding: 2em 2em 2em 0';>
            <b style='color: #ff4b4b'>During training</b>, we feed the convex combination to the critic so that it can evaluate the futre expected return of state action pairs <b style='color: #ff4b4b'>of a particular policy</b>. 
            <b style='color: #ff4b4b'>During evaluation</b> one can use this critic to get a good <b style='color: #ff4b4b'>0-shot approximation</b>  of the best policy of the subspace. The tool below captures both the reward and Q-value landscapes of tasks in the 4
            HalfCheetah scenarios (we enforced our model to get 4 anchors at the end of each scenario to visualize two interesting cases : 3 anchors, meaning that it has learned of the 3 first tasks, and 4 anchors, captured at the end of the whole scenario) and shows
            how well the Q function is able to estimate the reward landscape. Another possibility is simply given by rolling out trajectories in a <b style='color: #ff4b4b'>k-shot setting</b> (see figure on the right). It would be interesting as further work to try
            <b style='color: #ff4b4b'>active learning</b> methods during training to quickly get a good policy for a given task.
            </div>''', unsafe_allow_html = True)
        st.markdown('''
            <div style='text-align: justify; padding: 2em 2em 2em 0';>
            
            </div>''', unsafe_allow_html = True)
    with row_0_2:
        st.image("data/images/k_shot.gif")
    st.markdown("<hr>", unsafe_allow_html = True)
    row_1_1,row_1_2 = st.columns([1,3])
    with row_1_1:
        benchmark = st.selectbox('',["Forgetting_scenario","Transfer_scenario","Distraction_scenario","Composability_scenario"],index=1)
        task = st.selectbox('',range(len(scenarios[benchmark])),format_func=lambda x: str(x+1)+". "+scenarios[benchmark][x]+" task", index=3)
        n_anchors = st.radio('number of anchors:', (3,4), 1)
        landscape = st.radio('Landscape metric',("Reward","Q-value") if task+1 == n_anchors else ("Reward",))

    with row_1_2:
        d = data[benchmark][str(n_anchors)+"_anchors"][scenarios[benchmark][task]]
        alphas, rewards = d["alphas"].cpu(),d["q1" if landscape == "Q-value" else "rewards"].cpu()
        display_kshot(n_anchors,alphas,rewards,scenarios[benchmark], scenarios[benchmark][task])
        #st.markdown(selected_points)

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

if __name__ == "__main__":
    img = Image.open("data/images/icon.png")
    st.set_page_config(
        page_title="Continual Subspace of Policies",
        page_icon=img,
        layout="wide",
    )
    inject_custom_css('assets/styles.css')
    st.sidebar.markdown("<b>About</b>", unsafe_allow_html = True)
    st.sidebar.markdown("<div>Here you can visualize the reward landscape of the generated subspaces. For each scenario, we ran our model and evaluated numerous combinations to get an insight on the way new anchor are optimized.</div>", unsafe_allow_html = True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    run()