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

import dash
import dash_table

import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc


def frame_to_plot(df, metric_name):
	hover_columns = ["name","kernel","eps","k_l1","k_l2","gp_points","ivm_points"]
	plot_names = df["name"]
	metric_columns = []
	for c in list(df):
		if metric_name in c: #SMSE
			metric_columns.append(c)
	df[metric_name + "_mean"] = df[metric_columns].mean(axis=1)
	df[metric_name + "_var"] = df[metric_columns].var(axis=1)
	df[metric_name + "_std"] = df[metric_columns].std(axis=1)
	df["hover_text"] = df[hover_columns].apply(lambda x: '_'.join(x.map(str)), axis=1)
	# TODO df["color"]

	fig = {
		'data': [],
        'layout': dict(
            #yaxis = yaxis,
            xaxis = dict(
                categoryorder = "array",
                categoryarray = df["name"]
            ),
            showlegend=True,
            title = metric_name
        )
    }

	for t_name, t_df in df.groupby(["name"]): 
		fig['data'].append(
		    go.Bar(
		        #yaxis=dict(autorange=True),
		        #marker=dict(color=t_df["color"]),
		        x = t_df["hover_text"].values,
		        y = t_df[metric_name + "_mean"].values,
		        text = t_df["hover_text"].values,
		        error_y=dict(
		            type='data',
		            array = t_df[metric_name + "_std"].values,
		            visible=True
		        )
		    )
		)

	return fig

df = pd.read_csv("build/xval.csv")
df = df.round(4) # for displaying purposes everything is rounded

fig1 = frame_to_plot(df, "SMSE")
fig2 = frame_to_plot(df, "fit_time")

app = dash.Dash("Luxembourg experiments")

app.layout = html.Div([
	dash_table.DataTable(
	    id='table',
	    columns=[{"name": i, "id": i} for i in df.columns],
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
	        'maxHeight':'1000',
	        'maxWidth':'1800'
	    }
	),
	html.H6("SMSE"),
	dcc.Graph(id='plot-SMSE',figure=fig1),
	html.H6("FIT TIME"),
	dcc.Graph(id='plot-fittime',figure=fig2)
])

if __name__ == '__main__':
    app.run_server(debug=True)