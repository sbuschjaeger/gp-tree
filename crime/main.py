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

def sanitize(x, latex=False):
	if latex:
		ret_x = r"\text{" + str(x["name"]) + "}" + r"\\"
		if x["k_l2"] == "None":
			ret_x += str(x["kernel"]) + ", " + str(round(float(x["k_l1"]),2)) + r"\\"
		else:
			ret_x += str(x["kernel"]) + "," + str(round(float(x["k_l1"]),2)) + "," + str(round(float(x["k_l2"]),2)) + r"\\"
		
		ret_x += r"\varepsilon = " + str(x["eps"]) + r"\\"
		
		if x["gp_points"] != "None":
			ret_x += "c = " + str(x["gp_points"]) + r"\\"

		if x["ivm_points"] != "None":
			ret_x += r"\tau = " + str(x["ivm_points"])

		if "GMT" in ret_x:
			margin_text = ''.join(np.repeat(r"\phantom{+}\\", 2, axis=0))
			#print(margin_text)
		else:
			margin_text = ''.join(np.repeat(r"~\\", 2, axis=0))
			#print(margin_text)

		ret_x = "$" + margin_text + ret_x + "$"
		return ret_x
	else:
		ret_x = "" + str(x["name"]) + "<br>"
		if x["k_l2"] == "None":
			ret_x += str(x["kernel"]) + ", " + str(x["k_l1"]) + "<br>"
		else:
			ret_x += str(x["kernel"]) + "," + str(x["k_l1"]) + "," + str(x["k_l2"]) + "<br>"
		
		ret_x += "eps = " + str(x["eps"]) 
		
		if x["gp_points"] != "None":
			ret_x += ", c = " + str(x["gp_points"]) 
		if x["ivm_points"] != "None":
			ret_x += r", tau = " + str(x["ivm_points"]) 

		return ret_x
	# hover_columns = ["name","kernel","eps","k_l1","k_l2","gp_points","ivm_points"]

	# ret_x = ""
	# for c in hover_columns:
	# 	if x[c] == "None":
	# 		pass
	# 		#ret_x += "N "
	# 	elif (c == "name"):
	# 		ret_x += str(x[c]) + "\n"
	# 	elif c == "kernel":
	# 		ret_x += "k = "str(x[c]) + "\n"
	# 	elfi c == 
	# 	else:
	# 		if ()
	# 		ret_x += " = "str(x[c])
	# 		if (int(float(x[c])) == float(x[c])):
	# 			ret_x += str(int(float(x[c]))) + " "
	# 		else:
	# 			ret_x += str(round(float(x[c]),2)) + " "

	# return ret_x[:-1]

def frame_to_plot(param_df, metric_name, num_entries = 5, latex=False):
	# to avoid awkward problems with adding / removing columns, we will take a copy first
	df = param_df.copy()
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
	df["hover_text"] = df[hover_columns].apply(lambda x: sanitize(x,latex), axis=1)
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

		#if latex:
		t_name = r"$\text{" + t_name + "}$"
		#print("t_name = ", t_name)
		
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
				showgrid=True,
				zeroline=True,
				tickfont=dict(
		            size=16,
		            color='black',
		            family="Times New Roman,bold"
		        ),
			),
			showlegend=False,
			margin=dict(l=45, r=10, t=10, b=20)
		)
	}

	tmp_df = tmp_df.sort_values(metric_name + "_mean", ascending=True)
	box_plot = {
		'data': box_figure,
		'layout': dict(
			yaxis = dict(
				tickfont=dict(
		            size=16,
		            color='black',
		            family="Times New Roman,bold"
		        ),
			),
			xaxis = dict(
				categoryorder = "array",
				categoryarray = tmp_df["hover_text"],
				ticks = "outside",
			),
			showlegend=False,
			legend=dict(
				orientation="h",
	        	x=0.03, 
	        	y=0.99
			),
			margin=dict(l=35, r=10, t=10, b=90)
			#title = metric_name
		)
	}

	return box_plot, whisker_plot

df = pd.read_csv("build/xval.csv")
df = df.round(4) # for displaying purposes everything is rounded

experiment_name = "Crime" 

fig1, fig2 = frame_to_plot(df, "SMSE", 1)
fig1_latex, fig2_latex = frame_to_plot(df, "SMSE", 1, True)
pio.write_image(fig1_latex, experiment_name + '_SMSE_BOX.pdf')
pio.write_image(fig2_latex, experiment_name + '_SMSE_WHISKER.pdf')

fig3, fig4 = frame_to_plot(df, "MSE", 1)
fig3_latex, fig4_latex = frame_to_plot(df, "MSE", 1, True)
pio.write_image(fig3_latex, experiment_name + '_MSE_BOX.pdf')
pio.write_image(fig4_latex, experiment_name + '_MSE_WHISKER.pdf')

fig5, fig6 = frame_to_plot(df, "MAE", 1)
fig5_latex, fig6_latex = frame_to_plot(df, "MAE", 1, True)
pio.write_image(fig5_latex, experiment_name + '_MAE_BOX.pdf')
pio.write_image(fig6_latex, experiment_name + '_MAE_WHISKER.pdf')

fig7, fig8 = frame_to_plot(df, "fit_time", 1)
fig7_latex, fig8_latex = frame_to_plot(df, "fit_time", 1, True)
pio.write_image(fig7_latex, experiment_name + '_FITTIME_BOX.pdf')
pio.write_image(fig8_latex, experiment_name + '_FITTIME_WHISKER.pdf')

# fig4 = frame_to_plot(df, "fit_time", 5, False)
# pio.write_image(fig4, 'fig4.pdf')

app = dash.Dash(experiment_name + " experiments")

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