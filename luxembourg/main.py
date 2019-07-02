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

import plotly.io as pio

import dash
import dash_table

import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc

def simple_name(x):
	if "GMT-NN" in x:
		return "GMT-NN" 
	elif "GMT" in x:
		return "GMT"
	elif "GP" in x:
		return "GP"
	else:
		return "IVM"

def frame_to_plot(df, metric_name, num_entries = 10):
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
	df["simple_name"] = df["hover_text"].apply(lambda x: simple_name(x))

	data_figure = []
	tmp_df = None
	for t_name, t_df in df.groupby(["simple_name"]): 
		t_df = t_df.nsmallest(num_entries, metric_name + "_mean")
		if (tmp_df is None):
			tmp_df = t_df
		else:
			tmp_df = pd.concat([tmp_df, t_df])

		data_figure.append(
			go.Bar(
				#yaxis=dict(autorange=True),
				#marker=dict(color=t_df["color"]),
				x = t_df["hover_text"].values,
				y = t_df[metric_name + "_mean"].values,
				text = t_df["hover_text"].values,
				name = t_name,
				error_y=dict(
					type='data',
					array = t_df[metric_name + "_std"].values,
					visible=True
				)
			)
		)
	
	tmp_df = tmp_df.sort_values(metric_name + "_mean", ascending=False)
	fig = {
		'data': data_figure,
		'layout': dict(
			#yaxis = yaxis,
			xaxis = dict(
				categoryorder = "array",
				categoryarray = tmp_df["hover_text"]
			),
			showlegend=True,
			title = metric_name
		)
	}

	return fig

df = pd.read_csv("build/xval.csv")
df = df.round(4) # for displaying purposes everything is rounded

fig1 = frame_to_plot(df, "SMSE")
pio.write_image(fig1, 'fig1.pdf')

fig2 = frame_to_plot(df, "fit_time")
pio.write_image(fig2, 'fig2.pdf')

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
			'maxHeight':'600',
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