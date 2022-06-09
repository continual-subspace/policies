import streamlit as st
from utils import *
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def run_UI():
    inject_custom_css('assets/styles.css')
    d = {
        "Start":"At every stage during training, the subspace is <b style='color: #ff4b4b'>a simplex defined by a set of anchors</b> (i.e. vertices). Any policy (i.e. point) in this simplex can be represented as a convex combination α of the anchor parameters. αi defines the best policy in the subspace for task i.",
        "Grow":"When the agent encounters a new task, CSP tentatively <b style='color: #ff4b4b'>grows the subspace by adding a new anchor</b>.If the new task i is very different from previously seen ones, a better policy αnew can usually be learned in the new subspace. ",
        "Extend":"In this case, CSP <b style='color: #ff4b4b'>extends</b> the subspace by keeping the new anchor at the end of the training.",
        "Prune":"If the new task bear some similarities to previously seen ones, a good policy αold i can typically be found in the old subspace. In this case, CSP <b style='color: #ff4b4b'>prunes</b> the subspace by removing the new anchor.",
    }

    st.markdown("<h3 style='text-align: left';>Building a Subspace of Policies for scalable Continual Learning</h3>",unsafe_allow_html = True)
    with st.expander("Our Method",expanded=True):
        row_1_1,row_1_2 = st.columns([2,2])
        with row_1_2:
            state = st.select_slider(
                '',
                options=['Start', 'Grow', 'Extend', 'Prune'])
            image = Image.open('data/images/'+state+'.png')
            st.image(image, use_column_width = True)
            st.markdown("<div style='text-align: justify'>"+d[state]+"</div>", unsafe_allow_html = True)
        with row_1_1:
            st.markdown("""
            <div style='text-align: justify; padding: 2em 2em 2em 0';>
                Developing autonomous agents that can continuously acquire new knowledge and skills is a key open
                challenge in AI. This problem is referred to as continual reinforcement learning (CRL) and solving it
                is crucial for large-scale deployment of autonomous agents in non-stationary domains such as robotics
                or dialogue systems. The balance between 
                <span style="font:sans-serif; font-size:16px; color:#ff4b4b; font-weight:bold;">stability</span>,
                <span style="font:sans-serif; font-size:16px; color:#ff4b4b; font-weight:bold;">plasticity</span>, and
                <span style="font:sans-serif; font-size:16px; color:#ff4b4b; font-weight:bold;">scalability</span>
                is crucial for designing effective CRL methods. 
                While current methods perform well along some of these dimensions
                they tend to suffer along others. We take inspiration from the mode connectivity literature to develop a novel CRL
                method by iteratively learning a subspace of policies. <span style="font:sans-serif; font-size:16px; color:#ff4b4b; font-weight:bold;">Continual Subspace of Policies</span>
                (CSP) aims to strike a good balance between stability, plasticity, and scalability.
                Instead of learning a single policy, CSP maintains an entire subspace of policies defined as a convex
                hull in parameter space. The vertices of this convex hull are called anchors, with each anchor
                representing the parameters of a policy. This subspace captures a large number of diverse behaviors
                which enables efficient training on a wide range of tasks. 
            </div>    
                """,unsafe_allow_html = True)
            for _ in range(2):
                st.markdown("<div style='text-align: center'>"+"\n"+"</div>", unsafe_allow_html = True)
            
 
    with st.expander("Scalability"):
        row_2_1,row_2_2 = st.columns([4,2])
        with row_2_1:
            txt = """
            With CSP, training and inference wallclock times are similar to the ones when learning a single policy. Yet, the number
            of parameters is growing. We thus measure the scalability of our method with a <b style='color: #ff4b4b'>Growing Factor</b> metric.
            It is defined as the ratio between the number of parameters of the policy at last task and at first task. We made an ablation study
            by varying the number of tasks in a scenario (the composability one, on HalfCheetah) and measuring how much it grows compared to the naive - but strong - "fine-tune and clone"
            baseline that we call FT-N.
            <b style='color: #ff4b4b'>CSP maintains both strong performance and low memory cost</b> even as the number of tasks increases. 
            In contrast, FT-N’s growing factor scales linearly, which makes it impractical for long task sequences.
            """
            st.markdown("<div style='text-align: justify'>"+txt+"</div>", unsafe_allow_html = True)

        with row_2_2:
            _df = {
                "method":["CSP","CSP","CSP","FT-N","FT-N","FT-N","CSP","CSP","CSP","FT-N","FT-N","FT-N","CSP","CSP","CSP","FT-N","FT-N","FT-N"],
                "nb_tasks":[12,12,12,12,12,12,8,8,8,8,8,8,4,4,4,4,4,4],
                "perf":[1.38,1.52,1.4,1.41,1.42,1.38,1.45,1.45,1.42,1.6,1.44,1.41,1.49,1.37,1.25,1.34,1.41,1.74],
                "memory":[4,4,5,12,12,12,5,4,6,8,8,8,4.0,3.0,3.0,4,4,4]
            }
            df = pd.DataFrame(_df)
            sns.set_theme(style="darkgrid")
            plt.rcParams['savefig.facecolor']='#44aa9900'
            plt.rcParams['axes.facecolor']='#44aa9900'
            plt.rcParams['legend.facecolor']='white'
            #plt.rcParams['grid.color']='black'
            fig, ax = plt.subplots()
            sns.lineplot(ax = ax,x="nb_tasks",y="perf",style="method",data=_df,err_style="bars",ci=90, color="#44aa99",linewidth=5)
            sns.scatterplot(ax = ax, data=df[df["method"] == "FT-N"].groupby("nb_tasks").mean()["perf"],color="#44aa99", marker="o",s = 250)
            sns.scatterplot(ax = ax, data=df[df["method"] == "CSP"].groupby("nb_tasks").mean()["perf"],color="#44aa99", marker="o",s = 250)
            ax.legend(loc="lower right")
            ax.set_ylabel("Average Performance", labelpad=8.,fontsize=20,font = "Trebuchet MS")
            ax.set_ylim(0.,1.55)
            ax.set_xlabel("Number of Tasks", labelpad=11.,fontsize=20,font = "Trebuchet MS")
            ax2 = ax.twinx()
            sns.lineplot(x="nb_tasks",y="memory",style="method",data=_df,color="#ff4b4b",linewidth=5,ci=90,err_style="bars")
            sns.scatterplot(ax = ax2, data=df[df["method"] == "FT-N"].groupby("nb_tasks").mean()["memory"],color="#ff4b4b", marker="o",s = 250)
            sns.scatterplot(ax = ax2, data=df[df["method"] == "CSP"].groupby("nb_tasks").mean()["memory"],color="#ff4b4b", marker="o",s = 250)
            ax2.set_ylabel("Growing Factor",rotation=270,labelpad=25., fontsize=20,font = "Trebuchet MS")
            ax2.set_ylim(1,13)
            ax2.grid(False)
            ax2.legend(handles=[Line2D([], [], marker='_', color="#44aa99", label='Average Performance'), Line2D([], [], marker='_', color="#ff4b4b", label='Growing Factor')],loc="lower left")
            ax2.set_xticklabels(["",4,"","","",8,"","","",12], fontsize=20)
            plt.tight_layout()
            st.pyplot(fig)
    with st.expander("Plasticity"):
        row_3_1,row_3_2 = st.columns([4,2])
        with row_3_1:
            txt = """
            In continual learning, plasticity can been seen as the <b style='color: #ff4b4b'>ability of a system to acquire additional knowledge very quickly</b>. When growing, policies included in the subspace inherits from previous tasks knowledge, and can leverage these knowledge combinations to peform better on unseen tasks. 
            Indeed, as each anchor network acts like a "center of attraction", it induces a wide <b style='color: #ff4b4b'>functional diversity</b> in the subspace. 
            It is particularly striking when facing composable scenarios, where 2 tasks previously seen are combined. The figure on the right presents the reward landscape of the subspace
            on a HalfCheetah task that combines two particular variations of the environment (moon + tinyfeet). The upper anchor and the lower left anchor has respectively been trained on the  moon task and 
            the tinyfeet task. Therefore, it already contains a policy that reaches a <b style='color: #ff4b4b'>performance of 3</b> (i.e. 3 times better than a policy learned from scratch, with a budget of 1e6 interactions).
            """
            st.markdown("<div style='text-align: justify'>"+txt+"</div>", unsafe_allow_html = True)

        with row_3_2:
            st.image("data/images/plasticity.png")
    with st.expander("Stability"):
        row_4_1,row_4_2 = st.columns([1,1])
        
if __name__ == "__main__":
    img = Image.open("data/images/icon.png")
    st.set_page_config(
        page_title="Continual Subspace of Policies",
        page_icon=img,
        layout="wide",
    )
    st.sidebar.markdown("<b>About</b>", unsafe_allow_html = True)
    st.sidebar.markdown("<div>This website presents interactive demos of our paper.</div>", unsafe_allow_html = True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    run_UI()
