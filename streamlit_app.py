import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from matplotlib.patches import RegularPolygon
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import torch
import pickle
from datetime import datetime

PATH2 = os.getcwd()+"/data/halfcheetah_benchmark3"
COMMENT_TEMPLATE_MD = """{} - {}
> {}"""

def display_gridsearch(data,x,y,hue):
    fig, ax = plt.subplots(figsize = (8,8))
    sns.barplot(x=x, y=y, hue=hue, data=data,ax=ax)
    return fig

def display_kshot(alphas,values,value_name):
    n_anchors = alphas.shape[-1]
    fig, ax = plt.subplots(figsize = (16,12))
    plt.axis('off')
    if n_anchors == 2:
        ax = display_kshot_2anchors(fig,ax,alphas,values,value_name)
    elif n_anchors == 3:
        ax = display_kshot_3anchors(fig,ax,alphas,values,value_name)
    elif n_anchors == 4:
        fig = display_kshot_4anchors(fig,ax,alphas,values,value_name)
    return fig


def display_kshot_2anchors(fig,ax,alphas,rewards,label):
    plt.axis('off')
    n_anchors = alphas.shape[1]
    radius = 0.5
    center = (0.5,0.5)

    subspace = RegularPolygon((0.5,0.5),n_anchors,radius = radius, fc=(1,1,1,0), edgecolor="black")
    anchors = subspace.get_path().vertices[:-1] * radius + center

    for i,anchor in enumerate(anchors):
        x = anchor[0] -0.05 + (anchor[0]-center[0]) * 0.1
        y = anchor[1] + (anchor[1]-center[1]) * 0.2 if anchor[0]-center[0]!=0 else anchor[1] + (anchor[1]-center[1]) * 0.05
        ax.text(x,y,"("+"0,"*i+"1"+",0"*(n_anchors-i-1)+")",fontsize="x-large")

    coordinates = (alphas @ anchors).T
    ax.add_artist(subspace)
    points = ax.scatter(coordinates[0],coordinates[1],c=rewards, cmap="RdYlGn", s=45)
    x_best,y_best = coordinates[0][rewards.argmax()],coordinates[1][rewards.argmax()]
    
    best_point = ax.scatter(x_best,y_best, s=400, color="black", marker="x",linewidth=3, label='best reward')
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.,1.)
    ts = plt.text(x_best+0.05,y_best,str(int(rewards.max())),size=15)
    #plt.plot([x_best,projection[0] - 0.025 if x_best<0.5 else projection[0] + 0.045],[y_best,projection[1]],color="black",linewidth=2)
    cbar = fig.colorbar(points, ax=ax, pad=0.2, shrink = 0.5)
    #minVal = int(rewards.min().item())
    #maxVal = int(rewards.max().item())
    #cbar.set_ticks([minVal, maxVal])
    #cbar.set_ticklabels([minVal, maxVal])
    #ax.legend(handles=[best_point],loc="upper right", bbox_to_anchor=(0.7, 0.4, 0.7, 0.4))
    return ax

def display_kshot_3anchors(fig,ax,alphas,rewards,label):
    plt.axis('off')
    n_anchors = alphas.shape[1]
    radius = 0.5
    center = (0.5,0.5)

    subspace = RegularPolygon((0.5,0.5),n_anchors,radius = radius, fc=(1,1,1,0), edgecolor="black")
    anchors = subspace.get_path().vertices[:-1] * radius + center

    for i,anchor in enumerate(anchors):
        x = anchor[0] #-0.05 + (anchor[0]-center[0]) * 0.1
        y = anchor[1] #+ (anchor[1]-center[1]) * 0.2 if anchor[0]-center[0]!=0 else anchor[1] + (anchor[1]-center[1]) * 0.05
        ax.text(x,y,"θ"+str(i+1),fontsize="x-large")

    coordinates = (alphas @ anchors).T
    ax.add_artist(subspace)
    points = ax.scatter(coordinates[0],coordinates[1],c=rewards, cmap="RdYlGn", s=10)
    x_best,y_best = coordinates[0][rewards.argmax()],coordinates[1][rewards.argmax()]
    
    #projection
    p3 = np.array([x_best,y_best])
    p2 = np.array([0.5,1.])
    p1 = np.array([0.0669873, 0.25 ]) if x_best <= 0.5 else np.array([0.9330127, 0.25])
    l2 = np.sum((p1-p2)**2)
    t = np.sum((p3 - p1) * (p2 - p1)) / l2
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))
    projection = p1 + t * (p2 - p1)
    
    
    best_point = ax.scatter(x_best,y_best, s=90, facecolor='green', edgecolor="black", marker="p",linewidth=2, label='best reward')
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.,1.)
    ts = plt.text(projection[0] - 0.1 if x_best<0.5 else projection[0] + 0.05,projection[1],str(int(rewards.max())),size=15)
    plt.plot([x_best,projection[0] - 0.025 if x_best<0.5 else projection[0] + 0.045],[y_best,projection[1]],color="black",linewidth=1)
    cbar = fig.colorbar(points, ax=ax, pad=0.2, shrink = 0.5)
    #ax.legend(handles=[best_point],loc="upper right", bbox_to_anchor=(0.7, 0.4, 0.7, 0.4))
    return ax

scale=20
def display_kshot_4anchors(fig,ax,alphas,rewards,label):

    base = torch.Tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    x, y, z = (alphas @ base).T

    marker = {"size":4,"color":rewards,"colorscale":'RdYlGn',"opacity":0.2}
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode="markers", marker = marker,surfacecolor="white",name=label)])
    fig.update_traces(hovertemplate='%{marker.color:.0f}')

    line = {"color":"black","width":2}
    fig.add_trace(go.Scatter3d(x=[0,0,1,0,0,1,0,0], y=[0,1,0,0,0,0,0,1], z=[0,0,0,1,0,0,1,0], line = line, marker = {"size":0},name='subspace'))

    #best point
    x_best,y_best,z_best = x[rewards.argmax()].item(),y[rewards.argmax()].item(),z[rewards.argmax()].item()
    line = {"color":"black","dash":"dash","width":2}
    fig.add_trace(go.Scatter3d(x=[x_best,1], y=[y_best,0], z=[z_best,1], line = line, marker = {"size":[5,0],"color":"green","opacity":0.9}))

    t1,t2,t3,t4 = alphas[rewards.argmax()].tolist()
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="",showticklabels=False,showgrid=False,showbackground=False,showaxeslabels=False),
            yaxis=dict(title="",showticklabels=False,showgrid=False,showbackground=False,showaxeslabels=False),
            zaxis=dict(title="",showticklabels=False,showgrid=False,showbackground=False,showaxeslabels=False),
            annotations=[
            dict(showarrow=False,x=1,y=0,z=1,text="best "+label+"="+str(int(rewards.max().item())),xanchor="right",arrowcolor="green",xshift=-1,opacity=0.7),
            dict(showarrow=False,x=1,y=0,z=1,text="("+str(round(t1,1))+" , "+str(round(t2,1))+" , "+str(round(t3,1))+" , "+str(round(t4,1))+")",xanchor="right",arrowcolor="green",xshift=-1,yshift=-12,opacity=0.7,font={"size":10}),
            dict(showarrow=False,x=0,y=0,z=0,text="θ1",xanchor="left",xshift=10,opacity=0.7),
            dict(showarrow=False,x=1,y=0,z=0,text="θ2",xanchor="right",xshift=-10,opacity=0.7),
            dict(showarrow=False,x=0,y=1,z=0,text="θ3",xanchor="left",xshift=10,opacity=0.7),
            dict(showarrow=False,x=0,y=0,z=1,text="θ4",yshift=10,opacity=0.7)]),
        showlegend=False
    )
    return fig

def add_comment_section(comments,conn):

    with st.expander("💬 Open comments"):

        # Show comments

        st.write("**Comments:**")

        for index, entry in enumerate(comments.itertuples()):
            st.markdown(COMMENT_TEMPLATE_MD.format(entry.name, entry.date, entry.comment))

            is_last = index == len(comments) - 1
            is_new = "just_posted" in st.session_state and is_last
            if is_new:
                st.success("☝️ Your comment was successfully posted.")

        # Insert comment

        st.write("**Add your own comment:**")
        form = st.form("comment")
        name = form.text_input("Name")
        comment = form.text_area("Comment")
        submit = form.form_submit_button("Add comment")

        if submit:
            date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            db.insert(conn, [[name, comment, date]])
            if "just_posted" not in st.session_state:
                st.session_state["just_posted"] = True
            st.experimental_rerun()

def main():
    st.set_page_config(layout="wide")
    st.title('Building a Subspace of Policies for Scalable Continual Learning')
    """This page presents quantitative and qualitative results of the subspace method on 4 HalfCheetah scenarios."""

#########################
### Average perf eval ###
#########################
  #  st.markdown('##')
  #  st.subheader('Evaluation methods')
  #  row,_  = st.columns([1,1])
  #  with row:
  #      st.markdown("This chart presents the average performance (i.e. average cumulative rewards on all tasks) after each stage. It is split by evaluation methods. A description on evaluation methods is available on the right.")
  #  row_0_1,row_0_2,row_0_3  = st.columns([1,1,2])
  #  df = pd.read_csv(PATH2+"/gridsearch.csv")
  #  stop_words = ["avg","evaluation","iteration","id"]
  #  grid_hps = [col for col in df.columns if (sum([word in col for word in stop_words]) == 0)  and len(df[col].unique()) > 1]
  #  with row_0_1:
  #      x = st.selectbox('split',[None]+grid_hps,index=0)
  #      hue = st.selectbox('hue',[None]+grid_hps,index=0)
  #      method = st.selectbox('evaluation method',["oracle","value","last_anchor","midpoint","best_alpha"])
  #      stage = st.slider('average performance after stage:', 0, 3, 3)
  #  with row_0_2:
  #      y = method+"/avg_performance"
  #      df = df[df["iteration"] == stage]
  #      fig = display_gridsearch(df,x,y,hue)
  #      st.pyplot(fig)

#########################
### Average perf train ###
##########################
#    st.markdown('##')
#    st.subheader('Average performance by evaluation methods')
#    row,_  = st.columns([1,1])
#    with row:
#        st.markdown("This chart presents the average performance (average cumulative rewards on each task) after each stage. It is split by training methods (see *split* and *hue* on the left). Multiple *evaluation methods* are available.")
#    row_0_1,row_0_2,row_0_3  = st.columns([1,1,2])
#    df = pd.read_csv(PATH2+"/gridsearch.csv")
#    stop_words = ["avg","evaluation","iteration","id"]
#    grid_hps = [col for col in df.columns if (sum([word in col for word in stop_words]) == 0)  and len(df[col].unique()) > 1]
#    with row_0_1:
#        x = st.selectbox('split',[None]+grid_hps,index=0)
#        hue = st.selectbox('hue',[None]+grid_hps,index=0)
#        method = st.selectbox('evaluation method',["oracle","value","last_anchor","midpoint","best_alpha"])
#        stage = st.slider('average performance after stage:', 0, 3, 3)
#    with row_0_2:
#        y = method+"/avg_performance"
#        df = df[df["iteration"] == stage]
#        fig = display_gridsearch(df,x,y,hue)
#        st.pyplot(fig)
    

#############################
### Visualizing subspaces ###
#############################
    st.markdown('##')
    st.subheader('Visualizing the subspaces')
    st.write(
    """
    We empirically figured out that:
    - The subspace built during training is very diverse in terms of performance and **contains outperforming policies**.
    -  The reward landscape within the subspace is **smooth**.
    -  the Q function learned during training is somehow **able to approximate this landscape**. 
    -  A **natural structure arises from the subspace** when the training tasks are correlated: imagine you learn two different tasks A and B with a 2-anchors subspace (i.e. a line segment). To solve a task C that is both correlated with A and B, the middle of the subspace induce a natural pool of good policies for this task.
    """)
    st.markdown("The tool below enables to visualize the **reward and q estimation landscapes** of each task within the subspace. \
    To do so, we evaluated ≈8,000 different deterministic policies within the subspace after each stage. The estimation of the q function is averaged on the whole trajectory. \n \
    Results are averaged on 10 trajectories.")
    with st.expander("Visualize the subspaces"):
        row_2_1,row_2_2,row_2_3  = st.columns([1,2,2])
        with row_2_1:
            benchmark = st.selectbox('Scenario:',["1. Negative backward","3. Distraction","4. Compositionality"],index=2)
            path = os.getcwd()+"/data/halfcheetah_benchmark"+benchmark[0]+"/"
            seed = st.selectbox('Seed:',[int(d)for d in os.listdir(path) if os.path.isfile(path+d+"/eval.pkl")],index=0)
            stage = st.slider('Stage:', 1, 3, 3) 
            task = st.slider('Task:', 0, 3, 3)
            with open(path+str(seed)+"/eval.pkl", "rb") as f:
                data = pickle.load(f)["stage_"+str(stage)]["task_"+str(task)]
            alphas, rewards, q1 = data["alphas"],data["rewards"],data["values1"]
            cov = ((rewards - rewards.mean()) * (q1 - q1.mean())).sum() / (rewards.shape[0] - 1)
            std1 = rewards.std()
            std2 = q1.std()
            corr = round((cov / (std1 * std2)).item(),2)
            
        with row_2_2:
            st.markdown("<h6 style='text-align: center; color: black;'>Reward</h6>", unsafe_allow_html=True)
            fig = display_kshot(alphas,rewards,"reward")
            if stage == 3:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)

        with row_2_3:
            st.markdown("<h6 style='text-align: center; color: black;'>Q estimation</h6>", unsafe_allow_html=True)
            fig = display_kshot(alphas,q1,"q")
            if stage == 3:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)
        _,row_3_2 = st.columns([1,4])
        with row_3_2:
            st.markdown("<h4 style='text-align: center; color: black;'>Correlation = "+str(corr)+"</h4>", unsafe_allow_html=True)
    #conn = db.connect()
    #comments = db.collect(conn)
    #add_comment_section(comments,conn)
        

        
if __name__ == "__main__":
    main()
