import torch
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import RegularPolygon
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import plotly.graph_objects as go

clrmap = ['#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
        '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D',
        '#A50026'][::-1]
#clrmap = LinearSegmentedColormap.from_list("sunset", clrmap)
#clrmap.set_bad('#FFFFFF')


def inject_custom_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_current_route():
    try:
        return st.experimental_get_query_params()['nav'][0]
    except:
        return "about"

def display_gridsearch(data,x,y,hue):
    fig, ax = plt.subplots(figsize = (8,8))
    sns.barplot(x=x, y=y, hue=hue, data=data,ax=ax)
    return fig

def display_kshot(n_anchors,alphas,values,labels, task):
    if n_anchors == 3:
        fig = display_kshot_3anchors(alphas,values,labels[:3],task)
    elif n_anchors == 4:
        fig = display_kshot_4anchors(alphas,values,labels,task)
    st.plotly_chart(fig, use_container_width=True)
    #selected_points = plotly_events(fig, select_event=False, hover_event=False,override_height=800,override_width="100%")
    #if len(selected_points)>0:
    #    return selected_points[0]['pointNumber']
    #else:
    #    None


#def display_kshot_2anchors(fig,ax,alphas,values,label):
#    plt.axis('off')
#    n_anchors = alphas.shape[1]
#    radius = 0.5
#    center = (0.5,0.5)
#
#    subspace = RegularPolygon((0.5,0.5),n_anchors,radius = radius, fc=(1,1,1,0), edgecolor="black")
#    anchors = subspace.get_path().vertices[:-1] * radius + center
#
#    for i,anchor in enumerate(anchors):
#        x = anchor[0] -0.05 + (anchor[0]-center[0]) * 0.1
#        y = anchor[1] + (anchor[1]-center[1]) * 0.2 if anchor[0]-center[0]!=0 else anchor[1] + (anchor[1]-center[1]) * 0.05
#        ax.text(x,y,"("+"0,"*i+"1"+",0"*(n_anchors-i-1)+")",fontsize="x-large")
#
#    coordinates = (alphas @ anchors).T
#    ax.add_artist(subspace)
#    points = ax.scatter(coordinates[0],coordinates[1],c=values, cmap="RdYlGn", s=45)
#    x_best,y_best = coordinates[0][values.argmax()],coordinates[1][values.argmax()]
#    
#    best_point = ax.scatter(x_best,y_best, s=400, color="black", marker="x",linewidth=3, label='best reward')
#    ax.set_xlim(0.,1.)
#    ax.set_ylim(0.,1.)
#    ts = plt.text(x_best+0.05,y_best,str(int(values.max())),size=15)
#    #plt.plot([x_best,projection[0] - 0.025 if x_best<0.5 else projection[0] + 0.045],[y_best,projection[1]],color="black",linewidth=2)
#    cbar = fig.colorbar(points, ax=ax, pad=0.2, shrink = 0.5)
#    #minVal = int(values.min().item())
#    #maxVal = int(values.max().item())
#    #cbar.set_ticks([minVal, maxVal])
#    #cbar.set_ticklabels([minVal, maxVal])
#    #ax.legend(handles=[best_point],loc="upper right", bbox_to_anchor=(0.7, 0.4, 0.7, 0.4))
#    return ax

def makeAxis(title, tickangle):
    return {
      'title': title,
      'titlefont': { 'size': 20, 'family':'sans-serif'},
      'showline': True,
      'showgrid': True,
      'showticklabels':False,
      'linecolor':'black',
      'linewidth':1
      }

def display_kshot_3anchors(alphas,values,labels, task):
    layout = {"width":700, "height":700}
    max_val = round(values.max().item(),2)
    min_val = round(values.min().item(),2)
    font = { 'size': 18,'family':'sans-serif'}
    colorbar = dict(thickness=10,tickvals=[min_val,max_val],ticktext = [min_val,max_val],tickfont=font,x=0.95,y=0.5,len=1.06, )
    marker = {"symbol":"hexagon","size":5,"color":values,"colorscale":clrmap,"opacity":1.,'colorbar':colorbar}
    

    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': [i for i in map(lambda x: x[0], alphas)],
        'b': [i for i in map(lambda x: x[1], alphas)],
        'c': [i for i in map(lambda x: x[2], alphas)],
        'text': [i for i in map(lambda x: x, values)],
        'textfont': font,
        'line':{'width':2,'color':'black'},
        'marker': marker,  
    }),layout)
    t1,t2,t3 = alphas[values.argmax()].tolist()
    line = {"color":"white","width":3}
    fig.add_trace(go.Scatterternary(a=[t1],b=[t2],c=[t3], marker = {"symbol":"hexagon","size":6,'line':line,"color":[values.max().item()],"colorscale":['#364B9A','#364B9A'],"opacity":1.}))
    fig.update_traces(hoverlabel_font_family='sans-serif',
                      hoverlabel_font_color='black',
                      hovertemplate='%{marker.color:.2f}<br>(%{a:.2f},%{b:.2f},%{c:.2f})<extra></extra>')

    fig.update_layout({
        'showlegend':False,
        'ternary': {
            'aaxis': makeAxis(labels[0], 0),
            'baxis': makeAxis(labels[1], 45),
            'caxis': makeAxis(labels[2], -45)
        },
        'annotations': [dict(showarrow=False,x=0.5,y=1.2,text=task+' reward landscape',font = { 'size': 22,'family':'sans-serif'}),
                        #dict(showarrow=True,x=x_best,y=y_best,text="Perf(α*) = "+str(round(values.max().item(),2)),xanchor="right",arrowcolor="black",xshift=0.,yshift=0.,opacity=1., font = font),
                        #dict(showarrow=False,x=0.2,y=0.5,text="α* = ("+str(round(t1,2))+" , "+str(round(t2,2))+" , "+str(round(t3,2))+")",xanchor="right",arrowcolor="blue",xshift=0.,yshift=0.,opacity=1.,font={ 'size': 16,'family':'sans-serif'}),
        ]
    })
    return fig

def display_kshot_4anchors(alphas,values,labels, task):
    layout = {"width":800, "height":800}
    line = {"color":"black","dash":"dash","width":2}
    font = { 'size': 18,'family':'sans-serif','color':'black'}
    base = torch.Tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    x, y, z = (alphas @ base).T
    max_val = round(values.max().item(),2)
    min_val = round(values.min().item(),2)
    colorbar = dict(thickness=10,tickvals=[min_val+0.01,max_val-0.01],tickmode = 'array',ticktext = [str(min_val),str(max_val)],tickfont=font,x=0.9,y=0.6,len=0.62, )
    marker = {"size":5,"color":values,"colorscale":clrmap,"opacity":0.2,'colorbar':colorbar}
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode="markers", marker = marker,surfacecolor="white",)],layout = layout)
    fig.update_traces(hovertemplate='%{marker.color:.0f}')

    line = {"color":"black","width":2}
    fig.add_trace(go.Scatter3d(x=[0,0,1,0,0,1,0,0], y=[0,1,0,0,0,0,0,1], z=[0,0,0,1,0,0,1,0], line = line, marker = {"size":0},name='subspace'))

    #best point
    x_best,y_best,z_best = x[values.argmax()].item(),y[values.argmax()].item(),z[values.argmax()].item()

    hovertemplate = '%{marker.color:.2f}<br>(%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>'
    fig.add_trace(go.Scatter3d(x=[x_best,1], y=[y_best,0], z=[z_best,1], line = line, marker = {"size":[5,0],"color":"blue","opacity":0.9}))
    fig.update_traces(hovertemplate = hovertemplate)
    t1,t2,t3,t4 = alphas[values.argmax()].tolist()
    fig.update_layout(
        title = {
        'text': task+' reward landscape',
        'y':1.,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font':{ 'size': 22,'family':'sans-serif'}},
        scene=dict(
            
            xaxis=dict(title="",showticklabels=False,showgrid=False,showbackground=False,showaxeslabels=False),
            yaxis=dict(title="",showticklabels=False,showgrid=False,showbackground=False,showaxeslabels=False),
            zaxis=dict(title="",showticklabels=False,showgrid=False,showbackground=False,showaxeslabels=False),
            annotations=[
            dict(showarrow=False,x=1,y=0,z=1,text="Perf(α*) = "+str(round(values.max().item(),2)),xanchor="right",arrowcolor="black",xshift=-1,opacity=1., font = font),
            dict(showarrow=False,x=1,y=0,z=1,text="α* = ("+str(round(t1,2))+" , "+str(round(t2,2))+" , "+str(round(t3,2))+" , "+str(round(t4,2))+")",xanchor="right",arrowcolor="blue",xshift=-1,yshift=-17,opacity=1.,font={ 'size': 16,'family':'sans-serif'}),
            dict(showarrow=False,x=0,y=0,z=0,text=labels[0],xanchor="left",xshift=10,opacity=1., font=font),
            dict(showarrow=False,x=1,y=0,z=0,text=labels[1],xanchor="right",xshift=-10,opacity=1., font=font),
            dict(showarrow=False,x=0,y=1,z=0,text=labels[2],xanchor="left",xshift=10,opacity=1., font=font),
            dict(showarrow=False,x=0,y=0,z=1,text=labels[3],yshift=10,opacity=1., font=font)]),
        showlegend=False
    )
    return fig