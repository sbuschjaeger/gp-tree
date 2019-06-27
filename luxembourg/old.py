#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import datetime
import io
import json
import urllib
import sys

import numpy as np

import pandas as pd
from pandas.io.json import json_normalize  

import flask

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go

def column_class(num_cols):
    if num_cols == 1:
        return "twelve columns"
    elif num_cols == 2:
        return "six columns"
    else:
        return "three columns"

def toRaw(XS,YS,names,xName,yName,multi):
    csvStr = "name;" + xName + ";" + yName + "\n"

    for X,Y,n in zip(XS,YS,names):
        if multi:
            for x,y in zip(X,Y):
                csvStr += n+";"+x+";"+y+"\n"
        else:
            csvStr += str(n)+";"+str(Y)+"\n"

    return csvStr

def toBooktabs(XS,YS,names,xName,yName,multi):
    texStr = "\\begin{table}[]\n"
    if multi:
        texStr += "\\begin{tabular}{lll}\n"
        texStr += "\\hline\n"
        texStr += "name&" + xName + "&" + yName + "\n"
    else:
        texStr += "\\begin{tabular}{ll}\n"
        texStr += "\\hline\n"
        texStr += "name&" + xName + "&" + yName + "\n"

    for X,Y,n in zip(XS,YS,names):
        if multi:
            for x,y in zip(X,Y):
                texStr += n+"&"+x+"&"+y+"\\\\ \n" 
        else:
            texStr += n+"&"+Y+"\\\\ \n" 

    texStr += "\\bottomrule\n"
    texStr += "\\end{tabular}\n"
    texStr += "\\end{table}\n"
    return texStr

def toLineTikz(XS,YS,names,xName,yName):
    tikzStr = "\\begin{tikzpicture}\n"
    tikzStr += "\\begin{axis}[\n"
    tikzStr += "xlabel="+xName+",\n"
    tikzStr += "ylabel="+yName+"]\n"

    for X,Y,n in zip(XS,YS,names):
        tikzStr += "\\addplot[smooth] plot coordinates {\n"

        for x,y in zip(X,Y):
            tikzStr += "("+x+","+y+")\n"
        tikzStr += "};\n"
        tikzStr += "\\addlegendentry{"+n+"}\n"
    tikzStr += "\\end{axis}\n"
    tikzStr += "\\end{tikzpicture}\n"

    return tikzStr

def toBarTikz(XS,YS,names,xName,yName):
    tikzStr = "\\begin{tikzpicture}\n"
    tikzStr += "\\begin{axis}[\n"
    tikzStr += "ybar,\n"
    tikzStr += "ylabel={"+yName+"},\n"
    tikzStr += "xtick=data\n"
    tikzStr += "]\n"
    tikzStr += "\\addplot coordinates {\n"
    for x,y,n in zip(XS,YS,names):
        tikzStr += "("+n+","+y+") \n"
    tikzStr += "};\n"
    tikzStr += "\\end{axis}\n"
    tikzStr += "\\end{tikzpicture}\n"
    return tikzStr

def toBoxTikz(XS,YS,names,xName,yName):
    tikzStr = "\\begin{tikzpicture}\n"
    tikzStr += "\\begin{axis}[\n"
    tikzStr += "ybar,\n"
    tikzStr += "ylabel={"+yName+"},\n"
    tikzStr += "xtick=data\n"
    tikzStr += "]\n"
    tikzStr += "\\addplot coordinates {\n"
    for x,y,n in zip(XS,YS,names):
        tikzStr += "("+n+","+y+") \n"
    tikzStr += "};\n"
    tikzStr += "\\end{axis}\n"
    tikzStr += "\\end{tikzpicture}\n"
    return tikzStr

def generatePlotly(title, df, name_order,cutoff_name):
    header = list(df)

    #print("generate for ", plot_name)
    if not "x_name" in df: 
        minY = min([y-ys for y,ys in zip(df["mean"],df["std"])])
        maxY = max([y+ys for y,ys in zip(df["mean"],df["std"])])
        diff = maxY - minY
        yaxis = dict(
            autorange=False,
            range=[minY-0.1*diff, maxY+0.1*diff]
        )
            
        if name_order is None:
            name_order = df["model_name"]
        
        # Note: Per default plotly uses a stepsize of "1" per bar. Thus we can compute the correct position
        #       by using the index of the cutoff_name. Then by adding "+0.5" to that index we find place
        #       the line directly in the middle of the two relevant bars
        #       See: https://plot.ly/python/reference/#bar-dx
        cutoff_idx = np.where(name_order==cutoff_name)[0][0]
        fig = {
            'data': [],
            'layout': dict(
                yaxis = yaxis,
                xaxis = dict(
                    categoryorder = "array",
                    categoryarray = name_order
                ),
                showlegend=True,
                title = title,
                shapes = [
                    dict(
                        type = "line",
                        #x0 = cutoff_name,
                        x0 = cutoff_idx+0.5,
                        y0 = 0,
                        #x1 = cutoff_name,
                        x1 = cutoff_idx+0.5,
                        y1 = (maxY+0.2*maxY) if maxY > 0 else (minY+0.2*minY), # Since minY is negative, we need to add (minus * minus) the percentage
                        line = dict(
                            color = "black",
                            width = 2.5
                        )
                    )
                ]
            )
        }

        df["color"] = df.apply(lambda x: name_color(x['model_name']), axis=1)

        #print(df)
        
        # check if y_std contains any NaNs
        if (df["std"].isnull().values.any()):
            for t_name, t_df in df.groupby(["name"]): 
                fig['data'].append(
                    go.Bar(
                        #yaxis=dict(autorange=True),
                        marker=dict(color=t_df["color"]),
                        x = t_df["model_name"],
                        y = t_df["mean"],
                        name = t_name
                    )
                )
            # fig['data'].append(
            #   go.Bar(
            #       yaxis=dict(autorange=True),
            #       marker=dict(color=df["color"]),
            #       x = df["model_name"],
            #       y = df["y_mean"],
            #       name = title
            #   )
            # )
        else:
            #print(df["model_name"])
            for t_name, t_df in df.groupby(["name"]): 
                #print(t_df.head())
                fig['data'].append(
                    go.Bar(
                        #yaxis=dict(autorange=True),
                        marker=dict(color=t_df["color"]),
                        x = t_df["model_name"],
                        y = t_df["mean"],
                        name = t_name,
                        error_y=dict(
                            type='data',
                            array = t_df["std"],
                            visible=True
                        )
                    )
                )
            # fig['data'].append(
            #   go.Bar(
            #       marker=dict(color=df["color"]),
            #       x = df["model_name"],
            #       y = df["y_mean"],
            #       name = title,
            #       error_y=dict(
            #           type='data',
            #           array = df["y_std"],
            #           visible=True
            #       )
            #   )
            # )
    else:
        pass

        # fig = {'data':[], 
        #       'layout': go.Layout(
        #       xaxis={
        #           'title': plot.xname,
        #           'type': 'linear'
        #       },
        #       yaxis={
        #           'title': plot.yname,
        #           'type': 'linear'
        #       },
        #       margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
        #       height=450,
        #       hovermode='closest'
        #   )
        # }
        # if plot.YStdDev is None:
        #   for x,y,name in zip(plot.X,plot.Y,plot.names):
        #       fig['data'].append(
        #           go.Scatter(
        #               x=x,
        #               y=y,
        #               text=name,
        #               name=name,
        #               mode='markers+lines',
        #               marker={
        #                   'size': 15,
        #                   'opacity': 0.5
        #                   #'line': {'width': 0.5, 'color': 'white'}
        #               }
        #           )
        #       )
        # else:
        #   for x,y,name in zip(plot.X,plot.Y,plot.names):
        #       print("name = ", name)
        #       print("x = ", x)
        #       print("y = ", y)
        #       print("YStdDev = ", plot.YStdDev)
        #       print("xname = ", plot.xname)
        #       print("yname = ", plot.yname)
        #       fig['data'].append(
        #           go.Scatter(
        #               x=x,
        #               y=y,
        #               error_y=dict(
        #                   type='data',
        #                   array=plot.YStdDev,
        #                   visible=True
        #               ),
        #               text=name,
        #               name=name,
        #               mode='markers+lines',
        #               marker={
        #                   'size': 15,
        #                   'opacity': 0.5
        #                   #'line': {'width': 0.5, 'color': 'white'}
        #               }
        #           )
        #       )
    return fig


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = []
        df = None
        name = ""
        for contents, filename, date in zip(list_of_contents, list_of_names, list_of_dates):
            if (len(name) > 0):
                name += " / " + filename 
            else:
                name = filename

            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'json' in filename:
                    data = json.load(io.StringIO(decoded.decode('utf-8')))  
                    dff = json_normalize(data, sep="_")
                    dff = dff.replace(np.nan, '', regex=True)

                    if df is None:
                        df = dff
                    else:
                        df.append(dff)
            except Exception as e:
                print(e)
                # return html.Div([
                #   'There was an error processing this file. Please upload a valid JSON file. Error was: ' + str(e)
                # ])


        return html.Div([
            html.Div([
                html.Div([
                    html.H5(name + " - Last refreshed " + str(datetime.datetime.now())),
                    dash_table.DataTable(
                        id='datatable',
                        columns=[
                            {"name": i, "id": i, "deletable": False} for i in df.columns
                        ],
                        data=df.to_dict("rows"),
                        editable=False,
                        filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        row_deletable=False,
                        selected_rows=[],
                        style_table={
                            'overflowX':'auto',
                            'overflowY':'auto',
                            'maxHeight':'500',
                            'maxWidth':'1200'
                        }
                    )
                ], style={'width': '65%', 'display': 'inline-block'}),
                html.Div([
                    html.H5("Plotting options"),
                    html.H6("Sort by"),
                    dcc.Dropdown(
                        id='sort-options',
                        options=[
                            #{'label': 'Feature 1', 'value': 'F1'},
                            #{'label': 'Feature 2', 'value': 'F2'},
                        ],
                        value=''
                    ),
                    html.Div([], style={'padding': 5}),
                    html.H6("Maximum model size"),
                    dcc.Input(
                        id="max-model-size",
                        placeholder='Maximum model size',
                        type='text',
                        value=None
                    ),
                    html.H6("Maximum training time"),
                    dcc.Input(
                        id="max-train-time",
                        placeholder='Maximum training time',
                        type='text',
                        value=None
                    ),
                    html.H6("Minimum accuracy"),
                    dcc.Input(
                        id="min-accuracy",
                        placeholder='Minimum accuracy',
                        type='text',
                        value=None
                    ),
                    html.Div([], style={'padding': 5}),
                    html.H6("Plot"),
                    dcc.Checklist(
                        id='column-selection',
                        options= 
                        [
                            {'label': name_mapping[0][1], 'value': name_mapping[0][1]},
                            {'label': name_mapping[1][1], 'value': name_mapping[1][1]},
                            {'label': name_mapping[2][1], 'value': name_mapping[2][1]},
                            {'label': name_mapping[3][1], 'value': name_mapping[3][1]}
                        ],
                        values=[name_mapping[0][1],name_mapping[1][1],name_mapping[2][1],name_mapping[3][1]],
                        labelStyle={'display': 'inline-block'}
                    )
                ],style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'horizontal-align': 'left'})
            ]),
            html.Hr(),  # horizontal line
            html.Div(id="plot-holder")
        ])

@app.callback(
    Output('sort-options', 'options'),
    [Input('datatable', "derived_virtual_data"),
    Input('datatable', "data")])
def update_plot_features(rows,data):
    if (rows is None):
        rows = data

    df = pd.DataFrame(rows)
    df = getPlots(df)

    metrics = list(df.columns.levels[0])
    if "test_NSMSE" in metrics:
        sort_by = "test_NSMSE"
    else:
        sort_by = "test_Accuracy"

    check_options = []
    for h in metrics:
        if h != "name" and h != "model_name":
            check_options.append({'label':h, 'value':h})
    return check_options

@app.callback(
    Output('sort-options', 'value'),
    [Input('sort-options', "options")])
def update_plot_features(options):
    if "test_NSMSE" in options:
        return "test_NSMSE"
    else:
        return "test_Accuracy"

@app.callback(
    Output('plot-holder', 'children'),
    [Input('datatable', "derived_virtual_data"),
    Input('datatable', "data"),
    Input('sort-options', "value"),
    Input('column-selection', "values"),
    Input('max-model-size', "value"),
    Input('max-train-time', "value"),
    Input('min-accuracy', "value")])
def updateGraph(rows, data, sort_by, columns, max_model_size, max_train_time, min_accuracy):
    if (rows is None):
        rows = data

    df = pd.DataFrame(rows)
    df = getPlots(df)

    #print("BEGIN:", df["model_name"])
    df.loc[:,"name"] = df["model_name"].apply(lambda x: nice_name(x))

    def name_filter(name):
        for i in columns:
            if name == i:
                return 1
        return 0
    df = df[df["name"].apply(name_filter) != 0]

    if max_model_size is not None and max_model_size != "":
        max_model_size = float(max_model_size)
        df = df[df[("test_Size","mean")].apply(lambda x: float(x) < max_model_size) != 0]

    if max_train_time is not None and max_train_time != "":
        max_train_time = float(max_train_time)
        df = df[df[("fit_time","mean")].apply(lambda x: float(x) < max_train_time) != 0]

    if min_accuracy is not None and min_accuracy != "":
        min_accuracy = float(min_accuracy)
        df = df[df[("test_Accuracy","mean")].apply(lambda x: float(x) > min_accuracy) != 0]

    #print("dropped: ", set(df["model_name"].apply(row_filter)["model_name"]))
    #print(df["model_name"].apply(row_filter))
    #print(df.head())
    #print(df)
    #print(set(df["model_name"]))

    plotlyPlots = []
    divrow = []
    num_cols = 1

    header = list(df.columns.levels[0])
    p_threshold = 0.05
    cutoff_name = None

    tmp = df.sort_values([(sort_by,"mean")])
    ys = reversed(tmp[(sort_by,"mean")].values)
    ystd = reversed(tmp[(sort_by,"std")].values)
    ycnt = reversed(tmp[(sort_by,"count")].values)
    names = reversed(tmp["model_name"].values)
    best_y = None
    best_std = None
    best_count = None

    for y,s,c,name in zip(ys,ystd,ycnt,names):
        if best_y is None:
            best_y = y
            best_std = s
            best_count = c

        from scipy.stats import ttest_ind_from_stats
        t,p = ttest_ind_from_stats(best_y,best_std,best_count,y,s,c,equal_var=False)
        if p < p_threshold:
            cutoff_name = name
            break
    name_order = tmp["model_name"]
    #print("END:",df["model_name"])
    for name in header:
        dff = df[name]
        if name != "name" and name != "model_name":
            # if name == "test_Accuracy":
            #     #print(dff.head())
            #     #y =  dff["y"]
            #     dff.loc[:,"mean"] = dff.loc[:,"mean"] - min(dff.loc[:,"mean"])
            
            #print("name = ",name)
            pd.options.mode.chained_assignment = None  # default='warn'
            dff.loc[:,"name"] = df.loc[:,"name"]
            dff.loc[:,"model_name"] = df.loc[:,"model_name"]

            fig = generatePlotly(name,dff.copy(),name_order,cutoff_name)
            raw = "TODO"
            booktabs = "TODO"
            tikz = "TODO"

            if "x_name" in header:
                plotTitle = name + " over " + dff["x_name"][0]
            else:
                plotTitle = name

            divrow.append(
                html.Div([
                        html.H6(plotTitle),
                        dcc.Graph(id='plot-'+plotTitle,figure=fig),
                        html.A(
                            'Download Rawdata',
                            id='rawdata-plot-'+plotTitle,
                            download='rawdata-plot-'+plotTitle+".csv",
                            #href="",
                            href="data:text/txt;charset=utf-8," + urllib.parse.quote(raw),
                            target="_blank",
                            style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                        ), 
                        html.A(
                            'Download Booktabs table',
                            id='booktabs-'+plotTitle,
                            download='booktabs-'+plotTitle+".tex",
                            href="data:text/txt;charset=utf-8," + urllib.parse.quote(booktabs),
                            target="_blank",
                            style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                        ),
                        html.A(
                            'Download Tikz',
                            id='tikz-plot-'+plotTitle,
                            download='tikz-'+plotTitle+".tex",
                            href="data:text/txt;charset=utf-8," + urllib.parse.quote(tikz),
                            target="_blank",
                            style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                        )
                    ], 
                    id='plot-'+plotTitle+"-div",
                    className=column_class(num_cols)
                    #className="six columns"
                )
            )
            if (len(divrow) == num_cols):
                plotlyPlots.append(html.Div(divrow, className="row"))
                divrow = []

    paretoMetrics = ["test_Error","test_Size", "fit_time"]
    front_idx,coords = get_pareto_front(df,paretoMetrics)
    
    pdf = df[front_idx]
    num_cols = 2

    #print(list(pdf))
    # TODO WHY DO I NEED .values[0]???
    pdf["color"] = pdf.apply(lambda x: name_color(x['model_name'].values[0]), axis=1)
    #print(pdf.head())
    for m1 in paretoMetrics:
        for m2 in paretoMetrics:
            if m1 != m2:
                plotTitle = "Pareto Front " + m1 + " over " + m2

                fig = {
                    'data':[],
                    'layout':[]
                }

                otherMetric = None
                for m in paretoMetrics:
                    if (m != m1) and (m != m2):
                        otherMetric = m

                for t_name, t_df in pdf.groupby(["name"]): 
                    #print(t_df.head())
                    fig['data'].append(
                        go.Scatter(
                            #yaxis=dict(autorange=True),
                            marker=dict(
                                color=t_df["color"],
                                size=t_df[otherMetric]["mean"],
                                #size=size,
                                sizemode='area',
                                sizeref=2.*max(t_df[otherMetric]["mean"])/(20.**2),
                                sizemin=4
                            ),
                            x = t_df[m1]["mean"],
                            y = t_df[m2]["mean"],
                            name = t_name,
                            mode = 'markers'
                        )
                    )

                
                divrow.append(
                    html.Div([
                            html.H6(plotTitle),
                            dcc.Graph(id='plot-'+plotTitle,figure=fig),
                            html.A(
                                'Download Rawdata',
                                id='rawdata-plot-'+plotTitle,
                                download='rawdata-plot-'+plotTitle+".csv",
                                #href="",
                                href="data:text/txt;charset=utf-8," + urllib.parse.quote(raw),
                                target="_blank",
                                style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                            ), 
                            html.A(
                                'Download Booktabs table',
                                id='booktabs-'+plotTitle,
                                download='booktabs-'+plotTitle+".tex",
                                href="data:text/txt;charset=utf-8," + urllib.parse.quote(booktabs),
                                target="_blank",
                                style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                            ),
                            html.A(
                                'Download Tikz',
                                id='tikz-plot-'+plotTitle,
                                download='tikz-'+plotTitle+".tex",
                                href="data:text/txt;charset=utf-8," + urllib.parse.quote(tikz),
                                target="_blank",
                                style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                            )
                        ], 
                        id='plot-'+plotTitle+"-div",
                        className=column_class(num_cols)
                        #className="six columns"
                    )
                )
                if (len(divrow) == num_cols):
                    plotlyPlots.append(html.Div(divrow, className="row"))
                    divrow = []

    plotTitle = "Pareto Front"
    fig = {
        'data':[
            go.Scatter3d(
                x=df[front_idx]["test_Error"]["mean"],
                y=df[front_idx]["test_Size"]["mean"],
                z=df[front_idx]["fit_time"]["mean"],
                mode='markers',
                marker=dict(
                    size=5,
                    line=dict(
                        color='rgba(217, 217, 217, 0.14)',
                        width=0.5
                    ),
                    opacity=0.8
                )
            )
        ],
        'layout':[]
    }
    divrow.append(
        html.Div([
                html.H6("plotTitle"),
                dcc.Graph(id='plot-'+plotTitle,figure=fig),
                html.A(
                    'Download Rawdata',
                    id='rawdata-plot-'+plotTitle,
                    download='rawdata-plot-'+plotTitle+".csv",
                    #href="",
                    href="data:text/txt;charset=utf-8," + urllib.parse.quote(raw),
                    target="_blank",
                    style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                ), 
                html.A(
                    'Download Booktabs table',
                    id='booktabs-'+plotTitle,
                    download='booktabs-'+plotTitle+".tex",
                    href="data:text/txt;charset=utf-8," + urllib.parse.quote(booktabs),
                    target="_blank",
                    style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                ),
                html.A(
                    'Download Tikz',
                    id='tikz-plot-'+plotTitle,
                    download='tikz-'+plotTitle+".tex",
                    href="data:text/txt;charset=utf-8," + urllib.parse.quote(tikz),
                    target="_blank",
                    style={'marginLeft': '1.5em', 'marginRight': '1.5em'}
                )
            ], 
            id='plot-'+plotTitle+"-div",
            className=column_class(num_cols)
            #className="six columns"
        )
    )
    #print(df[front_idx]["model_name"])

    if (len(divrow) > 0):
        plotlyPlots.append(html.Div(divrow, className="row"))

    return plotlyPlots

def main(argv):
    app.title = 'Metrics explorer'
    app.run_server(debug=True)

if __name__ == '__main__':
    main(sys.argv[1:])