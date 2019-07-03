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

def sanitize(x):
	hover_columns = ["name","kernel","eps","k_l1","k_l2","gp_points","ivm_points"]

	ret_x = ""
	for c in hover_columns:
		if x[c] == "None":
			pass
			#ret_x += "N "
		elif (c == "name" or c == "kernel"):
			ret_x += str(x[c]) + " "
		else:
			if (int(float(x[c])) == float(x[c])):
				ret_x += str(int(float(x[c]))) + " "
			else:
				ret_x += str(round(float(x[c]),2)) + " "

	return ret_x[:-1]

def frame_to_plot(df, metric_name, num_entries = 5):
	hover_columns = ["name","kernel","eps","k_l1","k_l2","gp_points","ivm_points"]
	plot_names = df["name"]
	metric_columns = []
	for c in list(df):
		if metric_name in c: #SMSE
			metric_columns.append(c)
	df[metric_name + "_mean"] = df[metric_columns].mean(axis=1)
	df[metric_name + "_var"] = df[metric_columns].var(axis=1)
	df[metric_name + "_std"] = df[metric_columns].std(axis=1)
	#df["hover_text"] = df[hover_columns].apply(lambda x: '_'.join(x.map(str)), axis=1)
	df["hover_text"] = df[hover_columns].apply(lambda x: sanitize(x), axis=1)
	df["simple_name"] = df["hover_text"].apply(lambda x: simple_name(x))

	min_y = min(df[metric_name + "_mean"].values)
	max_y = max(df[metric_name + "_mean"].values)

	box_figure = []
	whisker_figure = []
	tmp_df = None
	for t_name, t_df in df.groupby(["simple_name"]):
		t_small = t_df.nsmallest(num_entries, metric_name + "_mean")
		t_large = t_df.nlargest(num_entries, metric_name + "_mean")
		t_both = pd.concat([t_small, t_large])

		if (tmp_df is None):
			tmp_df = t_both
		else:
			tmp_df = pd.concat([tmp_df, t_both])

		box_figure.append(
			go.Bar(
				#yaxis=dict(autorange=True),
				#marker=dict(color=t_df["color"]),
				x = t_both["hover_text"].values, #hover_text
				y = t_both[metric_name + "_mean"].values,
				text = t_both[metric_name + "_mean"].round(2).values, #t_both["hover_text"].values,
				name = t_name,
				textposition = 'outside'
			)
		)
		
		whisker_figure.append(
			go.Box(
				y=t_df[metric_name + "_mean"].values,
				name = t_name,
				boxpoints='all',
				jitter=0.5,
				whiskerwidth=0.2,
				marker=dict(
					size=4,
				),
				line=dict(width=1)
			)
		)

	whisker_plot = {
		'data': whisker_figure,
		'layout': dict(
			 yaxis=dict(
				#autorange=True,
				showgrid=True,
				zeroline=True,
				showline=True,
				tickfont=dict(
		            size=12,
		            color='black'
		        )
				#range=[0.48, 2.2],
				# tickvals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
				# ticktext = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2"]
			),
			xaxis = dict(
			categoryorder = "array",
			categoryarray = tmp_df["hover_text"] ,
			tickangle=-70,
				tickfont=dict(
					size=12,
					color='black'
				),
			),
			# xaxis = dict(
			#   categoryorder = "array",
			#   categoryarray = tmp_df["hover_text"] 
			# ), 
			showlegend=True,
			legend=dict(orientation="h"),
			margin=dict(l=45, r=10, t=10, b=0)
			#title = metric_name
		)
	}

	tmp_df = tmp_df.sort_values(metric_name + "_mean", ascending=True)
	box_plot = {
		'data': box_figure,
		'layout': dict(
			yaxis = dict(
				tickfont=dict(
		            size=12,
		            color='black'
		        )
			),
			xaxis = dict(
				categoryorder = "array",
				categoryarray = tmp_df["hover_text"] ,
				tickangle=-70,
				tickfont=dict(
		            size=12,
		            color='black'
		        ),
			),
			showlegend=True,
			legend=dict(
				orientation="h",
	        	x=0.03, 
	        	y=0.99
			),
			margin=dict(l=35, r=10, t=10, b=150)
			#title = metric_name
		)
	}

	return box_plot, whisker_plot

df = pd.read_csv("build/xval.csv")
df = df.round(4) # for displaying purposes everything is rounded

fig1, fig2 = frame_to_plot(df, "SMSE", 2)
pio.write_image(fig1, 'fig1.pdf')
pio.write_image(fig2, 'fig2.pdf')

fig3, fig4 = frame_to_plot(df, "MSE", 2)
pio.write_image(fig3, 'fig1.pdf')
pio.write_image(fig4, 'fig2.pdf')

fig5, fig6 = frame_to_plot(df, "MAE", 2)
pio.write_image(fig1, 'fig5.pdf')
pio.write_image(fig2, 'fig6.pdf')

fig7, fig8 = frame_to_plot(df, "fit_time", 2)
pio.write_image(fig3, 'fig7.pdf')
pio.write_image(fig4, 'fig8.pdf')

# fig4 = frame_to_plot(df, "fit_time", 5, False)
# pio.write_image(fig4, 'fig4.pdf')

app = dash.Dash("Housing experiments")

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
	html.H6("Top 3 best and worst SMSE"),
	dcc.Graph(id='plot-top-SMSE',figure=fig1),
	html.H6("Whisker SMSE Plot"),
	dcc.Graph(id='plot-whisker-SMSE',figure=fig2),
	html.H6("Top 3 best and worst MSE"),
	dcc.Graph(id='plot-best-mse',figure=fig3),
	html.H6("Whisker MSE Plot"),
	dcc.Graph(id='plot-whisker-mse',figure=fig4),
	html.H6("Top 3 best and worst MAE"),
	dcc.Graph(id='plot-best-mae',figure=fig5),
	html.H6("Whisker MAE Plot"),
	dcc.Graph(id='plot-whisker-mae',figure=fig6),
	html.H6("Best FIT TIME"),
	dcc.Graph(id='plot-best-fittime',figure=fig7),
	html.H6("Whisker FIT TIME Plot"),
	dcc.Graph(id='plot-whisker-fittime',figure=fig8)
])

if __name__ == '__main__':
	app.run_server(debug=True)