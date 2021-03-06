import streamlit as st
from utils import *
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams


def run_UI():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    inject_custom_css("assets/styles.css")
    d = {
        "Start": "At every stage during training, the subspace is <b style='color: #ff4b4b'>a simplex defined by a set of anchors</b> (i.e. vertices). Any policy (i.e. point) in this simplex can be represented as a convex combination α of the anchor parameters. αi defines the best policy in the subspace for task i.",
        "Grow": "When the agent encounters a new task, CSP tentatively <b style='color: #ff4b4b'>grows the subspace by adding a new anchor</b>.If the new task i is very different from previously seen ones, a better policy αnew can usually be learned in the new subspace. ",
        "Extend": "In this case, CSP <b style='color: #ff4b4b'>extends</b> the subspace by keeping the new anchor at the end of the training.",
        "Prune": "If the new task bear some similarities to previously seen ones, a good policy αold i can typically be found in the old subspace. In this case, CSP <b style='color: #ff4b4b'>prunes</b> the subspace by removing the new anchor.",
    }

    st.markdown(
        "<h3 style='text-align: left;';>Building a Subspace of Policies for scalable Continual Learning</h3>",
        unsafe_allow_html=True,
    )
    row_1_1, row_1_2 = st.columns([2, 2])
    with row_1_2:
        state = st.select_slider("", options=["Start", "Grow", "Extend", "Prune"])
        image = Image.open("data/images/" + state + ".png")
        st.image(image, use_column_width=True)
        st.markdown(
            "<div style='text-align: justify'>" + d[state] + "</div>",
            unsafe_allow_html=True,
        )
    with row_1_1:
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("Scalability"):

        txt = """
        With CSP, training and inference wallclock times are similar to the ones when learning a single policy. Yet, the number
        of parameters is growing. We thus measure the scalability of our method with a <b style='color: #ff4b4b'>Growing Factor</b> metric.
        It is defined as the ratio between the number of parameters of the policy at last task and at first task. We made an ablation study
        by varying the number of tasks in a scenario (the composability one, on HalfCheetah) and measuring how much it grows compared to the naive - but strong - "fine-tune and clone"
        baseline that we call FT-N.
        <b style='color: #ff4b4b'>CSP maintains both strong performance and low memory cost</b> even as the number of tasks increases. 
        In contrast, FT-N’s growing factor scales linearly, which makes it impractical for long task sequences.
        """
        st.markdown(
            "<div style='text-align: justify'>" + txt + "</div><br><br>",
            unsafe_allow_html=True,
        )

        _df = {
            "method": [
                "CSP",
                "CSP",
                "CSP",
                "FT-N",
                "FT-N",
                "FT-N",
                "CSP",
                "CSP",
                "CSP",
                "FT-N",
                "FT-N",
                "FT-N",
                "CSP",
                "CSP",
                "CSP",
                "FT-N",
                "FT-N",
                "FT-N",
            ],
            "nb_tasks": [12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
            "perf": [
                1.38,
                1.52,
                1.4,
                1.41,
                1.42,
                1.38,
                1.45,
                1.45,
                1.42,
                1.6,
                1.44,
                1.41,
                1.49,
                1.37,
                1.25,
                1.34,
                1.41,
                1.74,
            ],
            "memory": [4, 4, 5, 12, 12, 12, 5, 4, 6, 8, 8, 8, 4.0, 3.0, 3.0, 4, 4, 4],
        }
        df = pd.DataFrame(_df)
        sns.set_theme(style="darkgrid")
        plt.rcParams["savefig.facecolor"] = "#44aa9900"
        plt.rcParams["axes.facecolor"] = "#44aa9900"
        plt.rcParams["grid.color"] = ".8"
        plt.rcParams["axes.edgecolor"] = ".8"
        plt.rcParams["legend.facecolor"] = "white"
        # plt.rcParams['grid.color']='black'
        fig, ax = plt.subplots()
        sns.lineplot(
            ax=ax,
            x="nb_tasks",
            y="perf",
            style="method",
            data=_df,
            err_style="bars",
            ci=90,
            color="#44aa99",
            linewidth=5,
        )
        sns.scatterplot(
            ax=ax,
            data=df[df["method"] == "FT-N"].groupby("nb_tasks").mean()["perf"],
            color="#44aa99",
            marker="o",
            s=250,
        )
        sns.scatterplot(
            ax=ax,
            data=df[df["method"] == "CSP"].groupby("nb_tasks").mean()["perf"],
            color="#44aa99",
            marker="o",
            s=250,
        )
        ax.legend(loc="lower right")
        ax.set_ylabel(
            "Average Performance", labelpad=8.0, fontsize=14,
        )
        ax.set_ylim(0.0, 1.55)
        ax.set_xlabel("Number of Tasks", labelpad=11.0, fontsize=14)
        ax2 = ax.twinx()
        sns.lineplot(
            x="nb_tasks",
            y="memory",
            style="method",
            data=_df,
            color="#ff4b4b",
            linewidth=5,
            ci=90,
            err_style="bars",
        )
        sns.scatterplot(
            ax=ax2,
            data=df[df["method"] == "FT-N"].groupby("nb_tasks").mean()["memory"],
            color="#ff4b4b",
            marker="o",
            s=250,
        )
        sns.scatterplot(
            ax=ax2,
            data=df[df["method"] == "CSP"].groupby("nb_tasks").mean()["memory"],
            color="#ff4b4b",
            marker="o",
            s=250,
        )
        ax2.set_ylabel(
            "Growing Factor",
            rotation=270,
            labelpad=25.0,
            fontsize=14,
        )
        ax2.set_ylim(1, 13)
        ax2.grid(False)
        ax2.legend(
            handles=[
                Line2D(
                    [], [], marker="_", color="#44aa99", label="Average Performance"
                ),
                Line2D([], [], marker="_", color="#ff4b4b", label="Growing Factor"),
            ],
            loc="lower left",
        )
        ax2.set_xticklabels(["", 4, "", "", "", 8, "", "", "", 12], fontsize=20)
        plt.tight_layout()
        _, row_2_1, _ = st.columns([2, 3, 2])
        with row_2_1:
            st.pyplot(fig)
    with st.expander("Plasticity"):
        txt = """
        In continual learning, plasticity can been seen as the <b style='color: #ff4b4b'>ability of a system to acquire additional knowledge very quickly</b>. When growing, policies included in the subspace inherits from previous tasks knowledge, and can leverage these knowledge combinations to peform better on unseen tasks. 
        Indeed, as each anchor network acts like a "center of attraction", it induces a wide <b style='color: #ff4b4b'>functional diversity</b> in the subspace. 
        It is particularly striking when facing composable scenarios, where 2 tasks previously seen are combined. The figure below presents the reward landscape of the subspace
        on a HalfCheetah task that combines two particular variations of the environment (tinyfeet + moon). The upper anchor and the lower left anchor has respectively been trained on the  tinyfeet task and 
        the moon task. Interestingly, it already contains a policy that reaches a <b style='color: #ff4b4b'>performance of 2.98</b> (i.e. 3 times better than a policy learned from scratch with a budget of 1e6 interactions).
        """
        st.markdown(
            "<div style='text-align: justify'>" + txt + "</div><br><br>",
            unsafe_allow_html=True,
        )
        _, row_3_1, _ = st.columns([2, 3, 2])
        with row_3_1:
            st.image("data/images/plasticity.png")
    with st.expander("Stability"):

        txt = """
        Contrary to methods like EWC, CSP cannot suffer from <b style='color: #ff4b4b'>catastrophic forgetting</b>. Indeed, at the end of a training task
        the best policy is stored as a convex combination of the current subspace. At evaluation time, if the task id is given to the model,
        it uses this convex combination to instantiate an optimal policy and rollout a trajectory. The table below shows that - just as linear growing methods -
        <b style='color: #ff4b4b'>CSP has 0. forgetting</b>, while it is growing sublinearly (see scalability). Results are aggregated across 4 of our scenarios based on HalfCheetah, each consisting of a
        sequence of 8 tasks (see Designing Scenarios).
        """
        st.markdown(
            "<div style='text-align: justify'>" + txt + "</div><br><br>",
            unsafe_allow_html=True,
        )

        _, row_4_1, _ = st.columns([2, 3, 2])
        with row_4_1:
            txt = """
    | Method | Performance |Transfer|<b style='color: #ff4b4b'>Forgetting</b>   | Growing factor |
    |--------|-------------|-------------|--------------|----------------|
    | FT-1   | 0.75 ± 0.16 | 0.20 ± 0.14 |-0.45 ± 0.07|              1 |
    | FT-L2  | 0.81 ± 0.09 | 0.09 ± 0.14 |-0.28 ± 0.1|              2 |
    | EWC    | 0.98 ± 0.14 | 0.14 ± 0.11 |-0.28 ± 0.1|              3 |
    | PNN    | 1.06 ± 0.17 | 0.06 ± 0.18 |0.0 ± 0.0|           47.3 |
    | SAC-N  | 1.0 ± 0.00  | 0.0 ± 0.00  |0.0 ± 0.0|              8 |
    | FT-N   | 1.22 ± 0.1  | 0.22 ± 0.10 |0.0 ± 0.0|              8 |
    | <b style='color: #ff4b4b'>CSP</b> | 1.32 ± 0.07 | 0.31 ± 0.07 |<b style='color: #ff4b4b'>0.00 ± 0.0</b>|      4.0 ± 0.7 |
                    """
            st.markdown(txt, unsafe_allow_html=True)


if __name__ == "__main__":
    img = Image.open("data/images/icon.png")
    st.set_page_config(
        page_title="Continual Subspace of Policies",
        page_icon=img,
        layout="wide",
    )
    st.sidebar.markdown("<b>About</b>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div>This website presents main results and interactive demos of our paper.</div>",
        unsafe_allow_html=True,
    )
    st.set_option("deprecation.showPyplotGlobalUse", False)
    run_UI()
