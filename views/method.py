import streamlit as st
from PIL import Image

d = {
    "Start":"At every stage during training, the subspace is <b style='color: #ff4b4b'>a simplex defined by a set of anchors</b> (i.e. vertices). Any policy (i.e. point) in this simplex can be represented as a convex combination α of the anchor parameters. αi defines the best policy in the subspace for task i.",
    "Grow":"When the agent encounters a new task, CSP tentatively <b style='color: #ff4b4b'>grows the subspace by adding a new anchor</b>.If the new task i is very different from previously seen ones, a better policy αnew can usually be learned in the new subspace. ",
    "Extend":"In this case, CSP <b style='color: #ff4b4b'>extends</b> the subspace by keeping the new anchor.",
    "Prune":"If the new task bear some similarities to previously seen ones, a good policy αold i can typically be found in the old subspace. In this case, CSP <b style='color: #ff4b4b'>prunes</b> the subspace by removing the new anchor.",
}



def run():
    st.markdown("<h1 style='text-align: center';>Building a Subspace of Policies for scalable Continual Learning</h1>",unsafe_allow_html = True)
    #_,row_0_1,row_0_2,_  = st.columns([1,2,2,1])
    #with row_0_1:
    #    st.markdown("""
    #    <div style='text-align: justify';>
    #        Developing autonomous agents that can continuously acquire new knowledge and skills is a key open
    #        challenge in AI. This problem is referred to as continual reinforcement learning (CRL) and solving it
    #        is crucial for large-scale deployment of autonomous agents in non-stationary domains such as robotics
    #        or dialogue systems. The balance between stability (i.e. no forgetting of prior tasks), plasticity
    #        (i.e. positive transfer to new tasks including combinations of previously seen ones and the ability to21
    #        express many diverse behaviors), and scalability (i.e. memory increases sublinearly with the number22
    #        of tasks) is crucial for designing effective CRL methods. While current methods perform well along some of these dimensions
    #        they tend to suffer along others. We take inspiration from the mode connectivity literature to develop a novel CRL
    #        method by iteratively learning a subspace of policies. Continual Subspace
    #        Policies (CSP) aims to strike a good balance between stability, plasticity, and scalability.
    #        Instead of learning a single policy, CSP maintains an entire subspace of policies defined as a convex
    #        hull in parameter space. The vertices of this convex hull are called anchors, with each anchor
    #        representing the parameters of a policy. This subspace captures a large number of diverse behaviors38
    #        which enables efficient training on a wide range of tasks. 
    #    </div>    
    #        """,unsafe_allow_html = True)
    _,row_1_1,row_1_2,_  = st.columns([1,2,2,1])
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
    with row_1_2:
        state = st.select_slider(
            '',
            options=['Start', 'Grow', 'Extend', 'Prune'])
        image = Image.open('data/images/'+state+'.png')
        st.image(image, use_column_width = True)
        st.markdown("<div style='text-align: center'>"+d[state]+"</div>", unsafe_allow_html = True)

    with st.expander("Stability"):
        row_2_1,row_2_2 = st.columns([1,1])
    with st.expander("Plasticity"):
        row_2_1,row_2_2 = st.columns([1,1])
    with st.expander("Scalability"):
        row_2_1,row_2_2 = st.columns([1,1])
