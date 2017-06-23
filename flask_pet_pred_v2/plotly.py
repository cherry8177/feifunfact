
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.mlab as mlab
import math

def plotly_prob(mu, log_goal):
    
    plotly.tools.set_credentials_file\
    (username='cherry8177', api_key='3FTZrZRMQSDYshs8rxJ7')
    variance=0.71
    x= np.linspace(mu-5*variance,mu+5*variance, 100)
    x_new=[10**w for w in x ]
    y = mlab.normpdf(x, mu, (variance ** (1/2)))

    x1= np.linspace(log_goal, mu+5*variance, 100)
    x1_new=[10**w for w in x1 ]
    y1 = mlab.normpdf(x1, mu, (variance ** (1/2)))
    
    x2=[10**log_goal for _ in range(100)]
    y2= np.linspace(0, mlab.normpdf(log_goal, mu, variance ** (1/2)), 100)
    
    
    py.sign_in(username='cherry8177', api_key='3FTZrZRMQSDYshs8rxJ7')
    trace1 = {
      "x": x_new ,
      "y": y,

    "name": "PDF of Prediction Number", 
      "type": "scatter"
    }
    trace2 = {
      "x": x1_new,
      "y": y1,
      "fill": "tozeroy", 
      "name": "Prob(Predction> Your Goal)", 
      "type": "scatter"
    }
    trace3 = {
      "x": x2,
      "y": y2,
      
      "name": "Your Goal", 
      "type": "scatter"
    }

    layout = go.Layout(
        title='Your Chance of Reaching the Goal Number',
        xaxis=dict(
            title='Supporter Number',
            type='log',
        autorange=True,
            titlefont=dict(
                family='Courier New, monospace',
                size=24,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Probability',
            titlefont=dict(
                family='Courier New, monospace',
                size=24,
                color='#7f7f7f'
            )
        )
    )

    data = go.Data([trace1, trace2,trace3])

    fig = go.Figure(data=data, layout=layout)
    div = plotly.offline.plot(fig, show_link=False, output_type="div", include_plotlyjs=False)

    #py.iplot(fig)
    return div