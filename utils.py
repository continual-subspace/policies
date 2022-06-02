from matplotlib.patches import RegularPolygon
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go
import streamlit as st
import base64
from streamlit.components.v1 import html

NAVBAR_PATHS = {
    'ABOUT':'about',
    'DESIGNING SCENARIOS': 'scenarios',
    'VISUALIZING THE SUBSPACES': 'subspace'
}




def inject_custom_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_current_route():
    try:
        return st.experimental_get_query_params()['nav'][0]
    except:
        return "scenarios"


def navbar_component():
    navbar_items = ''
    for key, value in NAVBAR_PATHS.items():
        navbar_items += (f'<a class="navitem" href="https://share.streamlit.io/continual-subspace/policies/main/?nav={value}">{key}</a>')

    component = rf'''
            <nav class="container navbar" id="navbar">
                <ul class="navlist">
                {navbar_items}
                </ul>
            </nav>
            '''
    st.markdown(component, unsafe_allow_html=True)
    js = '''
    <script>
        // navbar elements
        var navigationTabs = window.parent.document.getElementsByClassName("navitem");
        var cleanNavbar = function(navigation_element) {
            navigation_element.removeAttribute('target')
        }
        
        for (var i = 0; i < navigationTabs.length; i++) {
            cleanNavbar(navigationTabs[i]);
        }
    </script>
    '''
    html(js)



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