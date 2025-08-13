
def make_sgd_plots():
    """Generates and saves three interactive plots demonstrating linear approximation, gradient descent, and Newton's method using Plotly."""
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    x = np.arange(-10,10,.1)
    y = x**2
    tangent = 8*x - 16

    y = y.tolist()
    tangent = tangent.tolist()

    fig1 = go.Figure(
        data = 
        [
            go.Scatter(x=x,y=y, name = '$f(x)=x^2$')
        ],
        layout = dict(
            xaxis = dict(
                tickmode = 'linear',
                tick0 = -10,
                dtick = 2,
                range = [-10,10]
            ),
            yaxis = dict(
                range = [-60,60])
        )
    )

    fig1.add_scatter(x = x, 
                    y = tangent, 
                    name = 'tangent at x=4',
                    line_color = 'crimson')

    for step in x:
        y = step**2
        yhat = 8*step -16
        SE = (yhat - y)**2
        fig1.add_scatter(
            mode = 'markers',
            x = [step],
            y = [yhat],
            line_color = 'red',
            visible = False, 
            showlegend = False,
            name = 'x = {step:.2f}'.format(step = step)) 
        
        fig1.add_scatter(
            x = [step,step],
            y = [yhat,y],
            mode = 'lines',
            visible = False,
            name = 'SE: {SE:.2f}'.format(SE = SE),
            showlegend = False,
            line = dict(
                color = 'black',
                width = 2,
                dash = 'dot'))

    fig1.data[0].visible = True
    fig1.data[1].visible = True
    fig1.data[202].visible = True
    fig1.data[202].showlegend = True
    fig1.data[203].visible = True
    fig1.data[203].showlegend = True

    steps = []
    for i in range(0,len(x)*2,2):
        step = dict(
            method = 'update',
            args = [{'visible': [True,True]+[False] * (len(x)*2),
                    'showlegend': [True,True]+[False]*(len(x)*2)}]
        )
        step['args'][0]['visible'][i] = True 
        step['args'][0]['visible'][i+1] = True 
        step['args'][0]['showlegend'][i] = True
        step['args'][0]['showlegend'][i+1] = True 
        steps.append(step)

    sliders = [dict(active = 90, 
                    steps = steps,
                    currentvalue = {'prefix': r'h: '}
                    )]

    fig1.layout.update(sliders = sliders,
                    updatemenus = [go.layout.Updatemenu(showactive = True)],
                    title = "Linear Approximation of f(x+h) from x = 4")

    x_array = np.array(x)
    steps_x = np.round(x_array-4.1,2).tolist()
    for i,step_x in enumerate(steps_x):
        fig1.layout.sliders[0].steps[i].label = step_x

    # Code for fig2
    def x_squared_grad(x,alpha):
        return x - (2*x*alpha)

    steps1 = [4]
    steps2 = [4]
    steps3 = [4]

    x1 = x2 = x3 = 4.0 
    for epoch in range(100):
        x1 = x_squared_grad(x1, .1)
        steps1.append(x1)
        x2 = x_squared_grad(x2, .01)
        steps2.append(x2)
        x3 = x_squared_grad(x3, .001)
        steps3.append(x3)

    x1 = np.array(steps1)
    y1 = x1**2
    x1 = x1.tolist()
    y1 = y1.tolist()
    x2 = np.array(steps2)
    y2 = x2**2
    x2 = x2.tolist()
    y2 = y2.tolist()
    x3 = np.array(steps3)
    y3 = x3**2
    x3 = x3.tolist()
    y3 = y3.tolist()

    frames = [dict(
        data=[
            go.Scatter(name = 'step: '+str(i),x= [x1[i]] ,y= [y1[i]],mode='markers',marker=dict(color='red',size=8),showlegend = True),
            go.Scatter(x= [x2[i]] ,y= [y2[i]],mode='markers',marker=dict(color='red',size=8),showlegend = False),
            go.Scatter(x= [x3[i]] ,y= [y3[i]],mode='markers',marker=dict(color='red',size=8),showlegend = False)
        ],
        traces = [0,1,2] 
    ) for i in range(len(x1))]

    fig2 = make_subplots(print_grid = True,
                        shared_xaxes = True,
                        shared_yaxes = True,
                        rows=1, cols=3, 
                        subplot_titles = ('Learning Rate: .1', 'Learning Rate: .01','Learning Rate: .001'))

    tangent_plot = go.Scatter(x = [4], 
                    y = [16], 
                    name = 'Starting Point at x=4',
                    marker = dict(color = 'black'),
                    mode = 'markers',
                    showlegend = False) 

    x_squared_plot = go.Scatter(x=x,y=[y], name = '$f(x)=x^2$',mode='lines',line=dict(width=2,color='blue'),showlegend=False) 

    global_minima = go.Scatter(showlegend = False,x=[-4,4],y = [0,0],mode='lines',line=dict(width=1,color='black'))

    fixed_traces = [tangent_plot,tangent_plot,tangent_plot, x_squared_plot,x_squared_plot,x_squared_plot,global_minima,global_minima,global_minima]

    fig2.add_traces(data = fixed_traces, rows = [1,1,1,1,1,1,1,1,1],cols=[1,2,3,1,2,3,1,2,3])

    fig2.data[0].showlegend = True
    fig2.data[3].showlegend = True

    layout = dict(
        xaxis = dict(
            tickmode = 'linear',
            autorange = False,
            tick0 = -10,
            dtick = 2,
            zeroline = False,
            range = [-10,10]
        ),
        yaxis = dict(
            autorange = False,
            zeroline = False,
            range = [-60,60]
        ),
        updatemenus=[dict(
            type = 'buttons',
            buttons=[
                dict(
                    label = 'Play',
                    method = 'animate',
                    args = [None]
                )
            ]
        )]
    )

    fig2.update_layout(layout,
                    title_text='Gradient Descent at Distinct Learning Rates')

    fig2.update(frames = frames)
    fig2.update_xaxes(range=[-10,10], dtick = 2)

    global_minima_annotations = (
        dict(
            x=0,
            xref='x',
            yref='y',
            y=0,
            text = 'Global Minima',
            ax=0,
            ay=-40,
            font = dict(size = 9),
            showarrow = True),
        dict(x=0,
            xref='x2',
            yref='y2',
            text = 'Global Minima',
            font = dict(size = 9),
            ax=0,
            ay=-40,
            showarrow = True),
        dict(x=0,
            xref='x3',
            yref='y3',
            font = dict(size = 9),
            text = 'Global Minima',
            ax=0,
            ay=-40,
            showarrow = True)
    )

    fig2.layout.annotations = fig2.layout.annotations + global_minima_annotations

    speed = [None, dict(frame = dict(duration = 300, redraw=True), 
                        transition = dict(duration = 0),
                        easing = 'linear', 
                        fromcurrent = True,
                        mode = 'immediate')]

    fig2.layout.updatemenus[0].buttons[0].args = speed

    # Code for fig3

    x = np.arange(-10,10,.1)
    y = x**2
    tangent = 8*x - 16

    x = x.tolist()
    y = y.tolist()
    tangent = tangent.tolist()

    fig3 = go.Figure(
        data = 
        [
            go.Scatter(x=x,y=y, name = '$f(x)=x^2$')
        ],
        layout = dict(
            xaxis = dict(
                tickmode = 'linear',
                tick0 = -10,
                dtick = 2,
                range = [-10,10]
            ),
            yaxis = dict(
                range = [-60,60])
        )
    )

    fig3.add_scatter(x = x, 
                    y = tangent, 
                    name = 'tangent at x=4',
                    line = dict(
                        color = 'crimson')
                    )

    fig3.add_scatter(x = [2,2],
                    y = [0,4],
                    name = 'New Displacement',
                    line = dict(
                        color = 'black')
                    )

    fig3.add_scatter(x = [4,2],
                    y = [16,0],
                    name = 'Newton\'s step at x = 4',
                    line = dict(
                        color = 'brown',
                        dash = 'dot')
                    )

    annotations = (dict(
        x=0,
        xref='x',
        yref='y',
        y=0,
        text = 'Global Minima',
        ax=0,
        ay=-40,
        font = dict(size = 12),
        showarrow = True),)

    fig3.layout.annotations = annotations
    fig3.layout.title = "One-Step Newton's Method"

    # Exporting the figures to HTML
    fig1.write_html("Jupyter_Notebooks/fig1.html")
    fig2.write_html("Jupyter_Notebooks/fig2.html")
    fig3.write_html("Jupyter_Notebooks/fig3.html")

def make_relu_hiplot_plots():
    """Generates and saves an interactive plot demonstrating the HitPlot using Plotly."""
    data = [{'activation': 'ReLU',
    'epoch': 1,
    'train_loss': 2.2567057985740937,
    'train_acc': 0.1966784381663113,
    'valid_loss': 2.20783610283574,
    'valid_acc': 0.11471518987341772},
    {'activation': 'ReLU',
    'epoch': 2,
    'train_loss': 1.9282052176339286,
    'train_acc': 0.3599413646055437,
    'valid_loss': 2.658291973645174,
    'valid_acc': 0.15090981012658228},
    {'activation': 'ReLU',
    'epoch': 3,
    'train_loss': 1.6841899164195762,
    'train_acc': 0.459005197228145,
    'valid_loss': 2.697213281559039,
    'valid_acc': 0.18067642405063292},
    {'activation': 'ReLU',
    'epoch': 4,
    'train_loss': 1.5303927748950559,
    'train_acc': 0.5022654584221748,
    'valid_loss': 3.5326503319076346,
    'valid_acc': 0.16050237341772153},
    {'activation': 'ReLU',
    'epoch': 5,
    'train_loss': 1.4438330806902986,
    'train_acc': 0.5197727878464818,
    'valid_loss': 3.8613895464547072,
    'valid_acc': 0.16416139240506328},
    {'activation': 'ReLU',
    'epoch': 6,
    'train_loss': 1.3931915998967217,
    'train_acc': 0.5263526119402985,
    'valid_loss': 4.698839404914953,
    'valid_acc': 0.16969936708860758},
    {'activation': 'ReLU',
    'epoch': 7,
    'train_loss': 1.358555124766791,
    'train_acc': 0.5311000799573561,
    'valid_loss': 5.268125509913964,
    'valid_acc': 0.16337025316455697},
    {'activation': 'ReLU',
    'epoch': 8,
    'train_loss': 1.3351253029634196,
    'train_acc': 0.5337653251599147,
    'valid_loss': 4.9521654346321204,
    'valid_acc': 0.14972310126582278},
    {'activation': 'ReLU',
    'epoch': 9,
    'train_loss': 1.3187274078824627,
    'train_acc': 0.5384295042643923,
    'valid_loss': 5.703450263301028,
    'valid_acc': 0.15130537974683544},
    {'activation': 'ReLU',
    'epoch': 10,
    'train_loss': 1.3039471396505198,
    'train_acc': 0.5426439232409381,
    'valid_loss': 5.4363789618769776,
    'valid_acc': 0.15644778481012658},
    {'activation': 'ReLU',
    'epoch': 11,
    'train_loss': 1.2915082008345549,
    'train_acc': 0.5424940031982942,
    'valid_loss': 5.915389435200751,
    'valid_acc': 0.15704113924050633},
    {'activation': 'ReLU',
    'epoch': 12,
    'train_loss': 1.2787634355427107,
    'train_acc': 0.5477078891257996,
    'valid_loss': 5.347536111179786,
    'valid_acc': 0.17543512658227847},
    {'activation': 'ReLU',
    'epoch': 13,
    'train_loss': 1.269061058060701,
    'train_acc': 0.5503231609808102,
    'valid_loss': 5.427630847013449,
    'valid_acc': 0.14487737341772153},
    {'activation': 'ReLU',
    'epoch': 14,
    'train_loss': 1.257962289903718,
    'train_acc': 0.5549706823027718,
    'valid_loss': 5.108685457253758,
    'valid_acc': 0.1597112341772152},
    {'activation': 'ReLU',
    'epoch': 15,
    'train_loss': 1.2495273354211087,
    'train_acc': 0.5575359808102346,
    'valid_loss': 5.108813322043117,
    'valid_acc': 0.15110759493670886},
    {'activation': 'ReLU',
    'epoch': 16,
    'train_loss': 1.2399636860341152,
    'train_acc': 0.5599680170575693,
    'valid_loss': 5.530833183964597,
    'valid_acc': 0.15634889240506328},
    {'activation': 'ReLU',
    'epoch': 17,
    'train_loss': 1.2324086008295576,
    'train_acc': 0.5630830223880597,
    'valid_loss': 5.8491543154173256,
    'valid_acc': 0.14883306962025317},
    {'activation': 'ReLU',
    'epoch': 18,
    'train_loss': 1.224833994786114,
    'train_acc': 0.5647654584221748,
    'valid_loss': 5.7223159210591374,
    'valid_acc': 0.16129351265822786},
    {'activation': 'ReLU',
    'epoch': 19,
    'train_loss': 1.2181698406683101,
    'train_acc': 0.5676972281449894,
    'valid_loss': 6.119402535354035,
    'valid_acc': 0.17375395569620253},
    {'activation': 'ReLU',
    'epoch': 20,
    'train_loss': 1.210808662463353,
    'train_acc': 0.5704957356076759,
    'valid_loss': 5.5784058389784414,
    'valid_acc': 0.17434731012658228},
    {'activation': 'ReLU',
    'epoch': 21,
    'train_loss': 1.2048782316098081,
    'train_acc': 0.572378065031983,
    'valid_loss': 5.527741637410997,
    'valid_acc': 0.150810917721519},
    {'activation': 'ReLU',
    'epoch': 22,
    'train_loss': 1.1980372186916977,
    'train_acc': 0.5757595948827292,
    'valid_loss': 5.488918256156052,
    'valid_acc': 0.17365506329113925},
    {'activation': 'ReLU',
    'epoch': 23,
    'train_loss': 1.193253962470016,
    'train_acc': 0.5757762526652452,
    'valid_loss': 5.657107968873616,
    'valid_acc': 0.16930379746835442},
    {'activation': 'ReLU',
    'epoch': 24,
    'train_loss': 1.190566984066831,
    'train_acc': 0.5796075426439232,
    'valid_loss': 5.608898694002176,
    'valid_acc': 0.16386471518987342},
    {'activation': 'ReLU',
    'epoch': 25,
    'train_loss': 1.1823643275669642,
    'train_acc': 0.5816897654584222,
    'valid_loss': 5.459799464744858,
    'valid_acc': 0.15822784810126583}]
    import hiplot as hip
    # Create an experiment from iterable data
    exp = hip.Experiment.from_iterable(data)

    # Save the experiment to an HTML file
    html_str = exp.to_html(force_full_width=False)

    # Write the HTML string to a file
    with open('../assets/html/relu_hiplot_graph.html', 'w') as f:
        f.write(html_str)


def make_tanh_relu_plots():
    """Generates and saves an interactive plot demonstrating the Tanh and ReLU activations using Plotly."""
    data = """---------------------------------------------------------------------------
Activation: tanh
Epoch: 01 | Epoch Time: 0m 23s
	Train Loss: 2.135 | Train Acc: 25.89%
	 Val. Loss: 2.144 |  Val. Acc: 16.62%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 02 | Epoch Time: 0m 38s
	Train Loss: 1.789 | Train Acc: 43.35%
	 Val. Loss: 2.489 |  Val. Acc: 15.23%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 03 | Epoch Time: 0m 39s
	Train Loss: 1.603 | Train Acc: 49.43%
	 Val. Loss: 3.176 |  Val. Acc: 16.60%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 04 | Epoch Time: 0m 39s
	Train Loss: 1.510 | Train Acc: 51.15%
	 Val. Loss: 3.610 |  Val. Acc: 16.60%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 05 | Epoch Time: 0m 33s
	Train Loss: 1.454 | Train Acc: 52.14%
	 Val. Loss: 3.997 |  Val. Acc: 17.10%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 06 | Epoch Time: 0m 17s
	Train Loss: 1.411 | Train Acc: 52.73%
	 Val. Loss: 4.143 |  Val. Acc: 17.32%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 07 | Epoch Time: 0m 18s
	Train Loss: 1.378 | Train Acc: 53.21%
	 Val. Loss: 4.446 |  Val. Acc: 16.69%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 08 | Epoch Time: 0m 17s
	Train Loss: 1.351 | Train Acc: 53.54%
	 Val. Loss: 4.687 |  Val. Acc: 16.29%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 09 | Epoch Time: 0m 17s
	Train Loss: 1.332 | Train Acc: 53.74%
	 Val. Loss: 4.419 |  Val. Acc: 16.83%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 10 | Epoch Time: 0m 17s
	Train Loss: 1.313 | Train Acc: 54.02%
	 Val. Loss: 4.609 |  Val. Acc: 16.77%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 11 | Epoch Time: 0m 17s
	Train Loss: 1.300 | Train Acc: 54.30%
	 Val. Loss: 4.505 |  Val. Acc: 16.28%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 12 | Epoch Time: 0m 17s
	Train Loss: 1.288 | Train Acc: 54.45%
	 Val. Loss: 4.570 |  Val. Acc: 15.96%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 13 | Epoch Time: 0m 17s
	Train Loss: 1.276 | Train Acc: 54.94%
	 Val. Loss: 4.893 |  Val. Acc: 15.96%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 14 | Epoch Time: 0m 17s
	Train Loss: 1.266 | Train Acc: 54.99%
	 Val. Loss: 4.865 |  Val. Acc: 17.41%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 15 | Epoch Time: 0m 17s
	Train Loss: 1.259 | Train Acc: 55.17%
	 Val. Loss: 4.627 |  Val. Acc: 11.21%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 16 | Epoch Time: 0m 17s
	Train Loss: 1.249 | Train Acc: 55.60%
	 Val. Loss: 4.820 |  Val. Acc: 18.63%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 17 | Epoch Time: 0m 18s
	Train Loss: 1.241 | Train Acc: 56.04%
	 Val. Loss: 4.849 |  Val. Acc: 16.69%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 18 | Epoch Time: 0m 18s
	Train Loss: 1.233 | Train Acc: 56.27%
	 Val. Loss: 4.656 |  Val. Acc: 17.83%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 19 | Epoch Time: 0m 17s
	Train Loss: 1.228 | Train Acc: 56.35%
	 Val. Loss: 4.678 |  Val. Acc: 17.43%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 20 | Epoch Time: 0m 17s
	Train Loss: 1.219 | Train Acc: 56.68%
	 Val. Loss: 4.487 |  Val. Acc: 18.07%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 21 | Epoch Time: 0m 18s
	Train Loss: 1.213 | Train Acc: 56.96%
	 Val. Loss: 4.870 |  Val. Acc: 17.18%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 22 | Epoch Time: 0m 17s
	Train Loss: 1.207 | Train Acc: 57.20%
	 Val. Loss: 4.697 |  Val. Acc: 18.75%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 23 | Epoch Time: 0m 17s
	Train Loss: 1.202 | Train Acc: 57.41%
	 Val. Loss: 4.862 |  Val. Acc: 16.95%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 24 | Epoch Time: 0m 17s
	Train Loss: 1.194 | Train Acc: 57.54%
	 Val. Loss: 4.773 |  Val. Acc: 17.14%
---------------------------------------------------------------------------
Activation: tanh
Epoch: 25 | Epoch Time: 0m 17s
	Train Loss: 1.189 | Train Acc: 57.83%
	 Val. Loss: 4.729 |  Val. Acc: 17.24%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 01 | Epoch Time: 0m 17s
	Train Loss: 2.229 | Train Acc: 19.80%
	 Val. Loss: 2.163 |  Val. Acc: 14.59%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 02 | Epoch Time: 0m 17s
	Train Loss: 1.891 | Train Acc: 35.63%
	 Val. Loss: 2.367 |  Val. Acc: 17.35%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 03 | Epoch Time: 0m 17s
	Train Loss: 1.675 | Train Acc: 45.36%
	 Val. Loss: 2.842 |  Val. Acc: 15.40%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 04 | Epoch Time: 0m 17s
	Train Loss: 1.546 | Train Acc: 50.31%
	 Val. Loss: 3.885 |  Val. Acc: 17.01%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 05 | Epoch Time: 0m 17s
	Train Loss: 1.461 | Train Acc: 51.76%
	 Val. Loss: 3.837 |  Val. Acc: 17.00%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 06 | Epoch Time: 0m 17s
	Train Loss: 1.407 | Train Acc: 52.52%
	 Val. Loss: 4.455 |  Val. Acc: 14.65%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 07 | Epoch Time: 0m 17s
	Train Loss: 1.371 | Train Acc: 53.03%
	 Val. Loss: 5.038 |  Val. Acc: 17.06%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 08 | Epoch Time: 0m 17s
	Train Loss: 1.346 | Train Acc: 53.43%
	 Val. Loss: 4.564 |  Val. Acc: 16.89%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 09 | Epoch Time: 0m 17s
	Train Loss: 1.324 | Train Acc: 53.94%
	 Val. Loss: 4.592 |  Val. Acc: 15.30%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 10 | Epoch Time: 0m 17s
	Train Loss: 1.308 | Train Acc: 54.31%
	 Val. Loss: 4.754 |  Val. Acc: 13.86%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 11 | Epoch Time: 0m 17s
	Train Loss: 1.294 | Train Acc: 54.36%
	 Val. Loss: 4.803 |  Val. Acc: 13.49%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 12 | Epoch Time: 0m 18s
	Train Loss: 1.282 | Train Acc: 54.94%
	 Val. Loss: 5.465 |  Val. Acc: 15.55%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 13 | Epoch Time: 0m 17s
	Train Loss: 1.270 | Train Acc: 55.12%
	 Val. Loss: 5.038 |  Val. Acc: 16.33%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 14 | Epoch Time: 0m 19s
	Train Loss: 1.260 | Train Acc: 55.39%
	 Val. Loss: 5.293 |  Val. Acc: 16.86%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 15 | Epoch Time: 0m 17s
	Train Loss: 1.251 | Train Acc: 55.74%
	 Val. Loss: 4.618 |  Val. Acc: 15.69%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 16 | Epoch Time: 0m 17s
	Train Loss: 1.238 | Train Acc: 56.18%
	 Val. Loss: 4.864 |  Val. Acc: 17.60%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 17 | Epoch Time: 0m 18s
	Train Loss: 1.228 | Train Acc: 56.52%
	 Val. Loss: 5.425 |  Val. Acc: 15.45%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 18 | Epoch Time: 0m 19s
	Train Loss: 1.219 | Train Acc: 56.80%
	 Val. Loss: 5.255 |  Val. Acc: 16.88%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 19 | Epoch Time: 0m 19s
	Train Loss: 1.211 | Train Acc: 57.14%
	 Val. Loss: 5.215 |  Val. Acc: 16.84%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 20 | Epoch Time: 0m 18s
	Train Loss: 1.204 | Train Acc: 57.29%
	 Val. Loss: 5.489 |  Val. Acc: 16.86%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 21 | Epoch Time: 0m 18s
	Train Loss: 1.197 | Train Acc: 57.66%
	 Val. Loss: 5.389 |  Val. Acc: 17.53%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 22 | Epoch Time: 0m 18s
	Train Loss: 1.190 | Train Acc: 57.75%
	 Val. Loss: 4.996 |  Val. Acc: 16.31%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 23 | Epoch Time: 0m 19s
	Train Loss: 1.185 | Train Acc: 58.14%
	 Val. Loss: 5.293 |  Val. Acc: 18.03%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 24 | Epoch Time: 0m 19s
	Train Loss: 1.180 | Train Acc: 58.35%
	 Val. Loss: 5.186 |  Val. Acc: 17.22%
---------------------------------------------------------------------------
Activation: leaky_relu
Epoch: 25 | Epoch Time: 0m 19s
	Train Loss: 1.178 | Train Acc: 58.29%
	 Val. Loss: 5.750 |  Val. Acc: 18.29%"""


    def parse_epochs(text):
        """
        Parse the given text into a list of dictionaries containing epoch information.

        Args:
            text (str): The text to be parsed.

        Returns:
            list: A list of dictionaries containing epoch information.
        """
        import re
        # Regular expression pattern to match epoch information
        pattern = r"Activation: (\w+)\s+Epoch: (\d+) \| Epoch Time: .*?\s+Train Loss: ([\d\.]+) \| Train Acc: ([\d\.]+)%\s+Val\. Loss: ([\d\.]+) \|  Val\. Acc: ([\d\.]+)%"

        # Find all matches in the given text
        matches = re.findall(pattern, text)

        # Create a list of dictionaries containing epoch information
        epochs = [
            {
                "activation": activation,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "valid_loss": float(val_loss),
                "valid_acc": float(val_acc),
            }
            for activation, epoch, train_loss, train_acc, val_loss, val_acc in matches
        ]

        return epochs

    epochs = parse_epochs(data)

    import hiplot as hip
    # Create an experiment from iterable data
    exp = hip.Experiment.from_iterable(epochs)

    # Save the experiment to an HTML file
    html_str = exp.to_html(force_full_width=False)

    # Write the HTML string to a file
    with open('../assets/html/relu_tanh_hiplot_graph.html', 'w') as f:
        f.write(html_str)

def make_linear_layer_plots():
    """Generates and saves an interactive plot demonstrating the linear layer using Plotly."""
    import hiplot as hip
    data = [{'epoch': 1.0,
    'train_loss': 2.212897131946295,
    'train_acc': 0.15091950959488273,
    'valid_loss': 11.462226964250396,
    'valid_acc': 0.09375},
    {'epoch': 2.0,
    'train_loss': 2.2014626053604744,
    'train_acc': 0.15766591151385928,
    'valid_loss': 15.43563340585443,
    'valid_acc': 0.0982001582278481},
    {'epoch': 3.0,
    'train_loss': 2.193212318013726,
    'train_acc': 0.15934834754797442,
    'valid_loss': 17.743525637856013,
    'valid_acc': 0.09464003164556962},
    {'epoch': 4.0,
    'train_loss': 2.1677922816164714,
    'train_acc': 0.1761560501066098,
    'valid_loss': 19.837977155854432,
    'valid_acc': 0.09681566455696203},
    {'epoch': 5.0,
    'train_loss': 2.1323169309701493,
    'train_acc': 0.1921808368869936,
    'valid_loss': 21.154041918018198,
    'valid_acc': 0.09473892405063292},
    {'epoch': 6.0,
    'train_loss': 2.100850640075293,
    'train_acc': 0.2055070628997868,
    'valid_loss': 21.467725536491297,
    'valid_acc': 0.09464003164556962},
    {'epoch': 7.0,
    'train_loss': 2.076701670567364,
    'train_acc': 0.2154517590618337,
    'valid_loss': 19.181373306467563,
    'valid_acc': 0.09543117088607594},
    {'epoch': 8.0,
    'train_loss': 2.0514450886610476,
    'train_acc': 0.22554637526652452,
    'valid_loss': 17.387509889240505,
    'valid_acc': 0.09642009493670886},
    {'epoch': 9.0,
    'train_loss': 2.0310485449426974,
    'train_acc': 0.22942763859275053,
    'valid_loss': 15.643752472310126,
    'valid_acc': 0.10225474683544304},
    {'epoch': 10.0,
    'train_loss': 2.012227853478145,
    'train_acc': 0.23962220149253732,
    'valid_loss': 15.169946598101266,
    'valid_acc': 0.09632120253164557},
    {'epoch': 11.0,
    'train_loss': 1.995873294659515,
    'train_acc': 0.24238739339019189,
    'valid_loss': 12.971168228342563,
    'valid_acc': 0.09918908227848101},
    {'epoch': 12.0,
    'train_loss': 1.9804057627598615,
    'train_acc': 0.2501998933901919,
    'valid_loss': 12.088009604924842,
    'valid_acc': 0.10265031645569621},
    {'epoch': 13.0,
    'train_loss': 1.967482056444896,
    'train_acc': 0.25264858742004265,
    'valid_loss': 11.534691919254351,
    'valid_acc': 0.10729825949367089},
    {'epoch': 14.0,
    'train_loss': 1.9549524107975746,
    'train_acc': 0.2572294776119403,
    'valid_loss': 9.970132175880142,
    'valid_acc': 0.09859572784810126},
    {'epoch': 15.0,
    'train_loss': 1.9429595882196162,
    'train_acc': 0.2642257462686567,
    'valid_loss': 10.950435590140428,
    'valid_acc': 0.10294699367088607},
    {'epoch': 16.0,
    'train_loss': 1.9351988835121268,
    'train_acc': 0.26690764925373134,
    'valid_loss': 9.349645687054984,
    'valid_acc': 0.1206487341772152},
    {'epoch': 17.0,
    'train_loss': 1.9280705238456157,
    'train_acc': 0.27137193496801704,
    'valid_loss': 9.406644797023338,
    'valid_acc': 0.1015625},
    {'epoch': 18.0,
    'train_loss': 1.9176410601845681,
    'train_acc': 0.2760361140724947,
    'valid_loss': 9.823314811609968,
    'valid_acc': 0.09859572784810126},
    {'epoch': 19.0,
    'train_loss': 1.9141617960004664,
    'train_acc': 0.27585287846481876,
    'valid_loss': 9.611549087717563,
    'valid_acc': 0.1027492088607595},
    {'epoch': 20.0,
    'train_loss': 1.9062367258295576,
    'train_acc': 0.2785014658848614,
    'valid_loss': 10.421080770371836,
    'valid_acc': 0.10403481012658228},
    {'epoch': 21.0,
    'train_loss': 1.902847127365405,
    'train_acc': 0.28058368869936035,
    'valid_loss': 10.30828007565269,
    'valid_acc': 0.10472705696202532},
    {'epoch': 22.0,
    'train_loss': 1.8937929718733342,
    'train_acc': 0.2863472814498934,
    'valid_loss': 9.669761174841772,
    'valid_acc': 0.10057357594936708},
    {'epoch': 23.0,
    'train_loss': 1.887595365804904,
    'train_acc': 0.28851279317697226,
    'valid_loss': 10.266508850870252,
    'valid_acc': 0.09948575949367089},
    {'epoch': 24.0,
    'train_loss': 1.8848772841984276,
    'train_acc': 0.28738006396588484,
    'valid_loss': 9.961499177956883,
    'valid_acc': 0.10067246835443038},
    {'epoch': 25.0,
    'train_loss': 1.8783976670775586,
    'train_acc': 0.29037846481876334,
    'valid_loss': 10.058255352551424,
    'valid_acc': 0.10106803797468354}]

    # Create an experiment from iterable data
    exp = hip.Experiment.from_iterable(data)

    # Save the experiment to an HTML file
    html_str = exp.to_html(force_full_width=True)

    # Write the HTML string to a file
    with open('../assets/linear/linear_layer_hiplot_graph.html', 'w') as f:
        f.write(html_str)

def make_logistic_plots():

    data = [{'epoch': 1.0,
        'train_loss': 0.6618835843842605,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6528506278991699,
        'valid_acc': 0.6328125},
        {'epoch': 2.0,
        'train_loss': 0.6584710745975889,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6494578719139099,
        'valid_acc': 0.6328125},
        {'epoch': 3.0,
        'train_loss': 0.6551205207561624,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6461004018783569,
        'valid_acc': 0.6328125},
        {'epoch': 4.0,
        'train_loss': 0.6518148882635708,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.642779529094696,
        'valid_acc': 0.6328125},
        {'epoch': 5.0,
        'train_loss': 0.6485495074041958,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6394957304000854,
        'valid_acc': 0.6328125},
        {'epoch': 6.0,
        'train_loss': 0.6453227339119747,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6362490057945251,
        'valid_acc': 0.6328125},
        {'epoch': 7.0,
        'train_loss': 0.6421338443098397,
        'train_acc': 0.6286945812807881,
        'valid_loss': 0.6330390572547913,
        'valid_acc': 0.6328125},
        {'epoch': 8.0,
        'train_loss': 0.6389825755152209,
        'train_acc': 0.6286945812807881,
        'valid_loss': 0.6298654079437256,
        'valid_acc': 0.6328125},
        {'epoch': 9.0,
        'train_loss': 0.6358680067391231,
        'train_acc': 0.6286945812807881,
        'valid_loss': 0.6267277598381042,
        'valid_acc': 0.640625},
        {'epoch': 10.0,
        'train_loss': 0.6327900722109038,
        'train_acc': 0.6286945812807881,
        'valid_loss': 0.623625636100769,
        'valid_acc': 0.640625},
        {'epoch': 11.0,
        'train_loss': 0.629748245765423,
        'train_acc': 0.6286945812807881,
        'valid_loss': 0.6205587387084961,
        'valid_acc': 0.640625},
        {'epoch': 12.0,
        'train_loss': 0.6267420670081829,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6175265908241272,
        'valid_acc': 0.640625},
        {'epoch': 13.0,
        'train_loss': 0.6237710097740436,
        'train_acc': 0.6265394088669951,
        'valid_loss': 0.6145287752151489,
        'valid_acc': 0.640625},
        {'epoch': 14.0,
        'train_loss': 0.6208349425217201,
        'train_acc': 0.624384236453202,
        'valid_loss': 0.6115648746490479,
        'valid_acc': 0.640625},
        {'epoch': 15.0,
        'train_loss': 0.6179331417741447,
        'train_acc': 0.624384236453202,
        'valid_loss': 0.6086344718933105,
        'valid_acc': 0.640625},
        {'epoch': 16.0,
        'train_loss': 0.6150652129074623,
        'train_acc': 0.624384236453202,
        'valid_loss': 0.6057370901107788,
        'valid_acc': 0.640625},
        {'epoch': 17.0,
        'train_loss': 0.6122307612978178,
        'train_acc': 0.6286945812807881,
        'valid_loss': 0.6028725504875183,
        'valid_acc': 0.640625},
        {'epoch': 18.0,
        'train_loss': 0.609429721174569,
        'train_acc': 0.6351600985221675,
        'valid_loss': 0.6000401973724365,
        'valid_acc': 0.640625},
        {'epoch': 19.0,
        'train_loss': 0.6066611059780779,
        'train_acc': 0.6394704433497537,
        'valid_loss': 0.5972397327423096,
        'valid_acc': 0.640625},
        {'epoch': 20.0,
        'train_loss': 0.603924718396417,
        'train_acc': 0.6394704433497537,
        'valid_loss': 0.5944706201553345,
        'valid_acc': 0.640625},
        {'epoch': 21.0,
        'train_loss': 0.6012202295763739,
        'train_acc': 0.645935960591133,
        'valid_loss': 0.5917326211929321,
        'valid_acc': 0.640625},
        {'epoch': 22.0,
        'train_loss': 0.598547179123451,
        'train_acc': 0.6480911330049262,
        'valid_loss': 0.5890252590179443,
        'valid_acc': 0.640625},
        {'epoch': 23.0,
        'train_loss': 0.5959050408725081,
        'train_acc': 0.6502463054187192,
        'valid_loss': 0.5863481163978577,
        'valid_acc': 0.640625},
        {'epoch': 24.0,
        'train_loss': 0.5932934201996902,
        'train_acc': 0.6502463054187192,
        'valid_loss': 0.5837007761001587,
        'valid_acc': 0.640625},
        {'epoch': 25.0,
        'train_loss': 0.5907119882517847,
        'train_acc': 0.6524014778325123,
        'valid_loss': 0.581082820892334,
        'valid_acc': 0.6484375},
        {'epoch': 26.0,
        'train_loss': 0.5881601530930092,
        'train_acc': 0.6545566502463054,
        'valid_loss': 0.5784939527511597,
        'valid_acc': 0.6484375},
        {'epoch': 27.0,
        'train_loss': 0.5856377831820784,
        'train_acc': 0.6588669950738917,
        'valid_loss': 0.5759336352348328,
        'valid_acc': 0.6484375},
        {'epoch': 28.0,
        'train_loss': 0.5831442865832098,
        'train_acc': 0.6631773399014779,
        'valid_loss': 0.5734015703201294,
        'valid_acc': 0.65625},
        {'epoch': 29.0,
        'train_loss': 0.580679202901906,
        'train_acc': 0.6674876847290641,
        'valid_loss': 0.5708974003791809,
        'valid_acc': 0.65625},
        {'epoch': 30.0,
        'train_loss': 0.5782424005968817,
        'train_acc': 0.6696428571428572,
        'valid_loss': 0.5684206485748291,
        'valid_acc': 0.65625},
        {'epoch': 31.0,
        'train_loss': 0.5758331561910695,
        'train_acc': 0.6696428571428572,
        'valid_loss': 0.5659710168838501,
        'valid_acc': 0.6640625},
        {'epoch': 32.0,
        'train_loss': 0.5734513381431843,
        'train_acc': 0.6739532019704434,
        'valid_loss': 0.5635480284690857,
        'valid_acc': 0.6875},
        {'epoch': 33.0,
        'train_loss': 0.5710964202880859,
        'train_acc': 0.6761083743842364,
        'valid_loss': 0.561151385307312,
        'valid_acc': 0.6875},
        {'epoch': 34.0,
        'train_loss': 0.5687679422312769,
        'train_acc': 0.6804187192118227,
        'valid_loss': 0.5587806701660156,
        'valid_acc': 0.6953125},
        {'epoch': 35.0,
        'train_loss': 0.5664656408901872,
        'train_acc': 0.6825738916256158,
        'valid_loss': 0.5564355850219727,
        'valid_acc': 0.6953125},
        {'epoch': 36.0,
        'train_loss': 0.5641893847235318,
        'train_acc': 0.6825738916256158,
        'valid_loss': 0.5541156530380249,
        'valid_acc': 0.6953125},
        {'epoch': 37.0,
        'train_loss': 0.5619383844836005,
        'train_acc': 0.686884236453202,
        'valid_loss': 0.5518207550048828,
        'valid_acc': 0.6953125},
        {'epoch': 38.0,
        'train_loss': 0.5597124428584658,
        'train_acc': 0.686884236453202,
        'valid_loss': 0.5495502352714539,
        'valid_acc': 0.6953125},
        {'epoch': 39.0,
        'train_loss': 0.5575110994536301,
        'train_acc': 0.6890394088669951,
        'valid_loss': 0.5473039746284485,
        'valid_acc': 0.6953125},
        {'epoch': 40.0,
        'train_loss': 0.5553342227278084,
        'train_acc': 0.6976600985221675,
        'valid_loss': 0.5450814962387085,
        'valid_acc': 0.6953125},
        {'epoch': 41.0,
        'train_loss': 0.5531812865158607,
        'train_acc': 0.708435960591133,
        'valid_loss': 0.5428824424743652,
        'valid_acc': 0.6953125},
        {'epoch': 42.0,
        'train_loss': 0.551051929079253,
        'train_acc': 0.7155172413793104,
        'valid_loss': 0.5407066345214844,
        'valid_acc': 0.6953125},
        {'epoch': 43.0,
        'train_loss': 0.5489459531060581,
        'train_acc': 0.7176724137931034,
        'valid_loss': 0.5385535955429077,
        'valid_acc': 0.703125},
        {'epoch': 44.0,
        'train_loss': 0.5468627995458143,
        'train_acc': 0.7176724137931034,
        'valid_loss': 0.5364230871200562,
        'valid_acc': 0.703125},
        {'epoch': 45.0,
        'train_loss': 0.5448023697425579,
        'train_acc': 0.7198275862068966,
        'valid_loss': 0.5343146920204163,
        'valid_acc': 0.703125},
        {'epoch': 46.0,
        'train_loss': 0.5427641704164702,
        'train_acc': 0.7219827586206896,
        'valid_loss': 0.5322281718254089,
        'valid_acc': 0.703125},
        {'epoch': 47.0,
        'train_loss': 0.5407478398290174,
        'train_acc': 0.7306034482758621,
        'valid_loss': 0.5301631689071655,
        'valid_acc': 0.7109375},
        {'epoch': 48.0,
        'train_loss': 0.538753345094878,
        'train_acc': 0.7306034482758621,
        'valid_loss': 0.5281195044517517,
        'valid_acc': 0.7109375},
        {'epoch': 49.0,
        'train_loss': 0.5367800942782698,
        'train_acc': 0.7306034482758621,
        'valid_loss': 0.5260966420173645,
        'valid_acc': 0.7109375},
        {'epoch': 50.0,
        'train_loss': 0.5348278900672649,
        'train_acc': 0.7349137931034483,
        'valid_loss': 0.5240945219993591,
        'valid_acc': 0.7109375},
        {'epoch': 51.0,
        'train_loss': 0.5328963049526872,
        'train_acc': 0.7370689655172413,
        'valid_loss': 0.5221127271652222,
        'valid_acc': 0.71875},
        {'epoch': 52.0,
        'train_loss': 0.5309851745079304,
        'train_acc': 0.7392241379310345,
        'valid_loss': 0.520150899887085,
        'valid_acc': 0.71875},
        {'epoch': 53.0,
        'train_loss': 0.5290941369944605,
        'train_acc': 0.7413793103448276,
        'valid_loss': 0.5182088613510132,
        'valid_acc': 0.71875},
        {'epoch': 54.0,
        'train_loss': 0.5272229951003502,
        'train_acc': 0.7413793103448276,
        'valid_loss': 0.516286313533783,
        'valid_acc': 0.7265625},
        {'epoch': 55.0,
        'train_loss': 0.5253712555457806,
        'train_acc': 0.7456896551724138,
        'valid_loss': 0.5143830180168152,
        'valid_acc': 0.734375},
        {'epoch': 56.0,
        'train_loss': 0.5235388525601091,
        'train_acc': 0.7456896551724138,
        'valid_loss': 0.5124986171722412,
        'valid_acc': 0.7421875},
        {'epoch': 57.0,
        'train_loss': 0.5217253586341595,
        'train_acc': 0.7478448275862069,
        'valid_loss': 0.5106328129768372,
        'valid_acc': 0.7421875},
        {'epoch': 58.0,
        'train_loss': 0.5199305764560042,
        'train_acc': 0.75,
        'valid_loss': 0.5087854862213135,
        'valid_acc': 0.7421875},
        {'epoch': 59.0,
        'train_loss': 0.5181542429430731,
        'train_acc': 0.7543103448275862,
        'valid_loss': 0.5069563388824463,
        'valid_acc': 0.7421875},
        {'epoch': 60.0,
        'train_loss': 0.5163960950127964,
        'train_acc': 0.7607758620689655,
        'valid_loss': 0.5051449537277222,
        'valid_acc': 0.7421875},
        {'epoch': 61.0,
        'train_loss': 0.5146557380413187,
        'train_acc': 0.7650862068965517,
        'valid_loss': 0.5033513307571411,
        'valid_acc': 0.7421875},
        {'epoch': 62.0,
        'train_loss': 0.512933139143319,
        'train_acc': 0.7672413793103449,
        'valid_loss': 0.5015749931335449,
        'valid_acc': 0.7578125},
        {'epoch': 63.0,
        'train_loss': 0.5112278379242996,
        'train_acc': 0.771551724137931,
        'valid_loss': 0.49981582164764404,
        'valid_acc': 0.7578125},
        {'epoch': 64.0,
        'train_loss': 0.509539768613618,
        'train_acc': 0.7737068965517241,
        'valid_loss': 0.4980735778808594,
        'valid_acc': 0.765625},
        {'epoch': 65.0,
        'train_loss': 0.5078684708167767,
        'train_acc': 0.7737068965517241,
        'valid_loss': 0.4963480234146118,
        'valid_acc': 0.765625},
        {'epoch': 66.0,
        'train_loss': 0.5062139116484543,
        'train_acc': 0.7737068965517241,
        'valid_loss': 0.4946388006210327,
        'valid_acc': 0.765625},
        {'epoch': 67.0,
        'train_loss': 0.5045756964847959,
        'train_acc': 0.7758620689655172,
        'valid_loss': 0.4929458796977997,
        'valid_acc': 0.765625},
        {'epoch': 68.0,
        'train_loss': 0.5029537924404802,
        'train_acc': 0.7758620689655172,
        'valid_loss': 0.49126893281936646,
        'valid_acc': 0.765625},
        {'epoch': 69.0,
        'train_loss': 0.501347673350367,
        'train_acc': 0.7801724137931034,
        'valid_loss': 0.48960769176483154,
        'valid_acc': 0.765625},
        {'epoch': 70.0,
        'train_loss': 0.4997573720997778,
        'train_acc': 0.7801724137931034,
        'valid_loss': 0.487962007522583,
        'valid_acc': 0.765625},
        {'epoch': 71.0,
        'train_loss': 0.49818255983549975,
        'train_acc': 0.7844827586206896,
        'valid_loss': 0.4863317012786865,
        'valid_acc': 0.765625},
        {'epoch': 72.0,
        'train_loss': 0.4966230063602842,
        'train_acc': 0.7844827586206896,
        'valid_loss': 0.4847164750099182,
        'valid_acc': 0.765625},
        {'epoch': 73.0,
        'train_loss': 0.4950785472475249,
        'train_acc': 0.7866379310344828,
        'valid_loss': 0.48311617970466614,
        'valid_acc': 0.765625},
        {'epoch': 74.0,
        'train_loss': 0.4935490509559368,
        'train_acc': 0.790948275862069,
        'valid_loss': 0.4815305769443512,
        'valid_acc': 0.765625},
        {'epoch': 75.0,
        'train_loss': 0.49203392554973735,
        'train_acc': 0.7931034482758621,
        'valid_loss': 0.47995948791503906,
        'valid_acc': 0.765625},
        {'epoch': 76.0,
        'train_loss': 0.49053353276746026,
        'train_acc': 0.7931034482758621,
        'valid_loss': 0.4784027636051178,
        'valid_acc': 0.765625},
        {'epoch': 77.0,
        'train_loss': 0.489047280673323,
        'train_acc': 0.7952586206896551,
        'valid_loss': 0.47686007618904114,
        'valid_acc': 0.765625},
        {'epoch': 78.0,
        'train_loss': 0.4875750706113618,
        'train_acc': 0.7974137931034483,
        'valid_loss': 0.4753313660621643,
        'valid_acc': 0.765625},
        {'epoch': 79.0,
        'train_loss': 0.4861166723843279,
        'train_acc': 0.8017241379310345,
        'valid_loss': 0.4738163650035858,
        'valid_acc': 0.765625},
        {'epoch': 80.0,
        'train_loss': 0.4846720531068999,
        'train_acc': 0.8017241379310345,
        'valid_loss': 0.4723149538040161,
        'valid_acc': 0.765625},
        {'epoch': 81.0,
        'train_loss': 0.4832408181552229,
        'train_acc': 0.8060344827586207,
        'valid_loss': 0.4708269238471985,
        'valid_acc': 0.765625},
        {'epoch': 82.0,
        'train_loss': 0.48182293464397563,
        'train_acc': 0.8103448275862069,
        'valid_loss': 0.4693520665168762,
        'valid_acc': 0.7734375},
        {'epoch': 83.0,
        'train_loss': 0.48041820526123047,
        'train_acc': 0.8146551724137931,
        'valid_loss': 0.4678902328014374,
        'valid_acc': 0.7734375},
        {'epoch': 84.0,
        'train_loss': 0.47902633403909617,
        'train_acc': 0.8168103448275862,
        'valid_loss': 0.46644127368927,
        'valid_acc': 0.78125},
        {'epoch': 85.0,
        'train_loss': 0.4776471894362877,
        'train_acc': 0.8168103448275862,
        'valid_loss': 0.4650050103664398,
        'valid_acc': 0.78125},
        {'epoch': 86.0,
        'train_loss': 0.4762807056821626,
        'train_acc': 0.8168103448275862,
        'valid_loss': 0.46358129382133484,
        'valid_acc': 0.78125},
        {'epoch': 87.0,
        'train_loss': 0.47492652103818694,
        'train_acc': 0.8168103448275862,
        'valid_loss': 0.4621698558330536,
        'valid_acc': 0.78125},
        {'epoch': 88.0,
        'train_loss': 0.47358470127500335,
        'train_acc': 0.8211206896551724,
        'valid_loss': 0.4607706367969513,
        'valid_acc': 0.7890625},
        {'epoch': 89.0,
        'train_loss': 0.4722549504247205,
        'train_acc': 0.8211206896551724,
        'valid_loss': 0.45938345789909363,
        'valid_acc': 0.7890625},
        {'epoch': 90.0,
        'train_loss': 0.4709371040607321,
        'train_acc': 0.8211206896551724,
        'valid_loss': 0.45800817012786865,
        'valid_acc': 0.7890625},
        {'epoch': 91.0,
        'train_loss': 0.46963099775643186,
        'train_acc': 0.8232758620689655,
        'valid_loss': 0.45664459466934204,
        'valid_acc': 0.7890625},
        {'epoch': 92.0,
        'train_loss': 0.468336532855856,
        'train_acc': 0.8275862068965517,
        'valid_loss': 0.45529258251190186,
        'valid_acc': 0.7890625},
        {'epoch': 93.0,
        'train_loss': 0.46705347916175577,
        'train_acc': 0.834051724137931,
        'valid_loss': 0.45395195484161377,
        'valid_acc': 0.796875},
        {'epoch': 94.0,
        'train_loss': 0.4657817709034887,
        'train_acc': 0.8362068965517241,
        'valid_loss': 0.452622652053833,
        'valid_acc': 0.8046875},
        {'epoch': 95.0,
        'train_loss': 0.4645211449984846,
        'train_acc': 0.8383620689655172,
        'valid_loss': 0.45130446553230286,
        'valid_acc': 0.8046875},
        {'epoch': 96.0,
        'train_loss': 0.4632716014467437,
        'train_acc': 0.8426724137931034,
        'valid_loss': 0.44999733567237854,
        'valid_acc': 0.8046875},
        {'epoch': 97.0,
        'train_loss': 0.46203294293633823,
        'train_acc': 0.8448275862068966,
        'valid_loss': 0.4487009644508362,
        'valid_acc': 0.8046875},
        {'epoch': 98.0,
        'train_loss': 0.4608049721553408,
        'train_acc': 0.8448275862068966,
        'valid_loss': 0.447415292263031,
        'valid_acc': 0.8046875},
        {'epoch': 99.0,
        'train_loss': 0.45958755756246633,
        'train_acc': 0.8448275862068966,
        'valid_loss': 0.4461402893066406,
        'valid_acc': 0.8046875},
        {'epoch': 100.0,
        'train_loss': 0.45838060050175106,
        'train_acc': 0.8448275862068966,
        'valid_loss': 0.44487571716308594,
        'valid_acc': 0.8046875},
        {'epoch': 101.0,
        'train_loss': 0.4571839036612675,
        'train_acc': 0.8448275862068966,
        'valid_loss': 0.443621426820755,
        'valid_acc': 0.8046875},
        {'epoch': 102.0,
        'train_loss': 0.4559974341556944,
        'train_acc': 0.8448275862068966,
        'valid_loss': 0.44237738847732544,
        'valid_acc': 0.8046875},
        {'epoch': 103.0,
        'train_loss': 0.45482102755842535,
        'train_acc': 0.8491379310344828,
        'valid_loss': 0.44114333391189575,
        'valid_acc': 0.8046875},
        {'epoch': 104.0,
        'train_loss': 0.45365445367221174,
        'train_acc': 0.8491379310344828,
        'valid_loss': 0.43991929292678833,
        'valid_acc': 0.8046875},
        {'epoch': 105.0,
        'train_loss': 0.45249781115301724,
        'train_acc': 0.8491379310344828,
        'valid_loss': 0.4387049674987793,
        'valid_acc': 0.8046875},
        {'epoch': 106.0,
        'train_loss': 0.45135073826230804,
        'train_acc': 0.8491379310344828,
        'valid_loss': 0.4375004470348358,
        'valid_acc': 0.8125},
        {'epoch': 107.0,
        'train_loss': 0.45021320211476296,
        'train_acc': 0.8491379310344828,
        'valid_loss': 0.4363054931163788,
        'valid_acc': 0.8125},
        {'epoch': 108.0,
        'train_loss': 0.4490851040544181,
        'train_acc': 0.8491379310344828,
        'valid_loss': 0.4351199269294739,
        'valid_acc': 0.8203125},
        {'epoch': 109.0,
        'train_loss': 0.4479663783106311,
        'train_acc': 0.8512931034482759,
        'valid_loss': 0.4339437484741211,
        'valid_acc': 0.8203125},
        {'epoch': 110.0,
        'train_loss': 0.4468568604567955,
        'train_acc': 0.8512931034482759,
        'valid_loss': 0.4327767789363861,
        'valid_acc': 0.8203125},
        {'epoch': 111.0,
        'train_loss': 0.4457563202956627,
        'train_acc': 0.8512931034482759,
        'valid_loss': 0.43161892890930176,
        'valid_acc': 0.8203125},
        {'epoch': 112.0,
        'train_loss': 0.4446647249419114,
        'train_acc': 0.853448275862069,
        'valid_loss': 0.4304701089859009,
        'valid_acc': 0.8203125},
        {'epoch': 113.0,
        'train_loss': 0.4435821072808627,
        'train_acc': 0.853448275862069,
        'valid_loss': 0.4293302297592163,
        'valid_acc': 0.8203125},
        {'epoch': 114.0,
        'train_loss': 0.442508105573983,
        'train_acc': 0.8599137931034483,
        'valid_loss': 0.42819908261299133,
        'valid_acc': 0.8203125},
        {'epoch': 115.0,
        'train_loss': 0.44144278559191474,
        'train_acc': 0.8599137931034483,
        'valid_loss': 0.42707669734954834,
        'valid_acc': 0.8203125},
        {'epoch': 116.0,
        'train_loss': 0.44038595002273034,
        'train_acc': 0.8599137931034483,
        'valid_loss': 0.4259628653526306,
        'valid_acc': 0.8203125},
        {'epoch': 117.0,
        'train_loss': 0.43933756598110857,
        'train_acc': 0.8599137931034483,
        'valid_loss': 0.424857497215271,
        'valid_acc': 0.8203125},
        {'epoch': 118.0,
        'train_loss': 0.43829746904044314,
        'train_acc': 0.8620689655172413,
        'valid_loss': 0.4237605333328247,
        'valid_acc': 0.8203125},
        {'epoch': 119.0,
        'train_loss': 0.4372657249713766,
        'train_acc': 0.8642241379310345,
        'valid_loss': 0.4226718544960022,
        'valid_acc': 0.8984375},
        {'epoch': 120.0,
        'train_loss': 0.43624200492069637,
        'train_acc': 0.8642241379310345,
        'valid_loss': 0.4215913414955139,
        'valid_acc': 0.8984375},
        {'epoch': 121.0,
        'train_loss': 0.43522624311776,
        'train_acc': 0.8642241379310345,
        'valid_loss': 0.4205189347267151,
        'valid_acc': 0.8984375},
        {'epoch': 122.0,
        'train_loss': 0.4342184724478886,
        'train_acc': 0.8663793103448276,
        'valid_loss': 0.41945451498031616,
        'valid_acc': 0.8984375},
        {'epoch': 123.0,
        'train_loss': 0.43321859425511855,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.41839802265167236,
        'valid_acc': 0.8984375},
        {'epoch': 124.0,
        'train_loss': 0.4322263454568797,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.41734933853149414,
        'valid_acc': 0.8984375},
        {'epoch': 125.0,
        'train_loss': 0.43124179182381467,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.4163084030151367,
        'valid_acc': 0.8984375},
        {'epoch': 126.0,
        'train_loss': 0.43026470315867454,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.41527506709098816,
        'valid_acc': 0.8984375},
        {'epoch': 127.0,
        'train_loss': 0.42929527677338697,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.41424936056137085,
        'valid_acc': 0.90625},
        {'epoch': 128.0,
        'train_loss': 0.42833305227345436,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.4132310152053833,
        'valid_acc': 0.90625},
        {'epoch': 129.0,
        'train_loss': 0.4273781940854829,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.4122201204299927,
        'valid_acc': 0.90625},
        {'epoch': 130.0,
        'train_loss': 0.4264304391269026,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.41121649742126465,
        'valid_acc': 0.90625},
        {'epoch': 131.0,
        'train_loss': 0.4254899189389985,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.4102201461791992,
        'valid_acc': 0.90625},
        {'epoch': 132.0,
        'train_loss': 0.42455633755387934,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.4092308580875397,
        'valid_acc': 0.90625},
        {'epoch': 133.0,
        'train_loss': 0.42362982651283,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.4082486629486084,
        'valid_acc': 0.90625},
        {'epoch': 134.0,
        'train_loss': 0.4227101556186018,
        'train_acc': 0.8685344827586207,
        'valid_loss': 0.40727338194847107,
        'valid_acc': 0.90625},
        {'epoch': 135.0,
        'train_loss': 0.4217972919858735,
        'train_acc': 0.8706896551724138,
        'valid_loss': 0.4063050448894501,
        'valid_acc': 0.90625},
        {'epoch': 136.0,
        'train_loss': 0.4208910711880388,
        'train_acc': 0.8728448275862069,
        'valid_loss': 0.40534353256225586,
        'valid_acc': 0.90625},
        {'epoch': 137.0,
        'train_loss': 0.41999155899574014,
        'train_acc': 0.8728448275862069,
        'valid_loss': 0.40438878536224365,
        'valid_acc': 0.90625},
        {'epoch': 138.0,
        'train_loss': 0.41909852521172886,
        'train_acc': 0.8728448275862069,
        'valid_loss': 0.4034406840801239,
        'valid_acc': 0.90625},
        {'epoch': 139.0,
        'train_loss': 0.41821210137728987,
        'train_acc': 0.8728448275862069,
        'valid_loss': 0.4024991989135742,
        'valid_acc': 0.90625},
        {'epoch': 140.0,
        'train_loss': 0.4173319586392107,
        'train_acc': 0.8728448275862069,
        'valid_loss': 0.40156421065330505,
        'valid_acc': 0.90625},
        {'epoch': 141.0,
        'train_loss': 0.41645826142409753,
        'train_acc': 0.875,
        'valid_loss': 0.4006357192993164,
        'valid_acc': 0.90625},
        {'epoch': 142.0,
        'train_loss': 0.4155908781906654,
        'train_acc': 0.875,
        'valid_loss': 0.39971357583999634,
        'valid_acc': 0.90625},
        {'epoch': 143.0,
        'train_loss': 0.4147295458563443,
        'train_acc': 0.875,
        'valid_loss': 0.39879775047302246,
        'valid_acc': 0.90625},
        {'epoch': 144.0,
        'train_loss': 0.4138744617330617,
        'train_acc': 0.875,
        'valid_loss': 0.39788818359375,
        'valid_acc': 0.90625},
        {'epoch': 145.0,
        'train_loss': 0.41302542850889007,
        'train_acc': 0.875,
        'valid_loss': 0.39698484539985657,
        'valid_acc': 0.90625},
        {'epoch': 146.0,
        'train_loss': 0.4121824461838295,
        'train_acc': 0.875,
        'valid_loss': 0.3960876166820526,
        'valid_acc': 0.90625},
        {'epoch': 147.0,
        'train_loss': 0.4113452845606311,
        'train_acc': 0.875,
        'valid_loss': 0.3951963782310486,
        'valid_acc': 0.90625},
        {'epoch': 148.0,
        'train_loss': 0.4105140422952586,
        'train_acc': 0.875,
        'valid_loss': 0.3943111300468445,
        'valid_acc': 0.90625},
        {'epoch': 149.0,
        'train_loss': 0.40968858784642714,
        'train_acc': 0.875,
        'valid_loss': 0.3934318423271179,
        'valid_acc': 0.90625},
        {'epoch': 150.0,
        'train_loss': 0.40886885544349405,
        'train_acc': 0.875,
        'valid_loss': 0.39255842566490173,
        'valid_acc': 0.90625}]

    # Create an experiment from iterable data
    import hiplot as hip
    exp = hip.Experiment.from_iterable(data)

    # Save the experiment to an HTML file
    html_str = exp.to_html(force_full_width=True)

    # Write the HTML string to a file
    with open('../assets/logistic/logistic_training_hiplot.html', 'w') as f:
        f.write(html_str)


if __name__ == "__main__":
    make_logistic_plots()