import re
import pickle

from sqlalchemy import text
import pandas as pd
from stqdm import stqdm
from transformers import pipeline
import torch
import plotly.graph_objects as go
import streamlit as st

import settings
import queries as nq
from postgre import postgre_engine as engine


@st.cache_data
def get_data():
    with engine.connect() as conn:
        fetch = conn.execute(text(nq.FindAllFromJobplanetReview)).fetchall()

    jp_df = pd.DataFrame(fetch)
    jp_df['all_text'] = jp_df['pros']+jp_df['cons']+jp_df['to_managements']
    filename = ['jp_comp_name_list']
    comp_name_ls = tuple(pickle.load(open(filename[0], 'rb')))
    return jp_df, comp_name_ls


def divide_into_chunks(text, max_chunk_size):
    num_chunks = (len(text) - 1) // max_chunk_size + 1
    chunk_indices = [i * max_chunk_size for i in range(num_chunks)] + [len(text)]
    return [text[chunk_indices[i]:chunk_indices[i+1]] for i in range(num_chunks)]


def preprocess_result_text(x):
    pattern = r"\[.*?\]"
    return eval(re.search(pattern, x['choices'][0]['text']).group())


def get_model():
    model = pipeline("zero-shot-classification", model=settings.model_name)
    return model


@st.cache_data
def get_result(_model, docs, candidate_labels, multi_label_input, idx, sample_n):
    multi_label = True if multi_label_input == "ON" else False
    outputs = []
    for doc in stqdm(docs[int(idx):int(idx)+sample_n]):
        output = _model(doc, candidate_labels, multi_label=multi_label)
        outputs.append(output)
    result = pd.DataFrame(outputs)

    return result[['sequence', 'labels', 'scores']]


@st.cache_data
def get_df_concat(reviews_chunks, candidate_labels, sample_n):
    model = get_model()
    df_concat = pd.DataFrame()
    for i in stqdm(range(0, len(candidate_labels), 2)):
        result = get_result(model, reviews_chunks, candidate_labels[i:i+2], 'OFF', 0, sample_n).rename(columns={"labels": f"labels_{i}", "scores": f"scores_{i}"})
        if not df_concat.empty:
            df_concat = pd.concat([df_concat, result.iloc[:,1:]], axis=1)
        else:
            df_concat = result
    return df_concat


@st.cache_resource
def draw_radar_chart(values, categories, user, name, user_nm, comp_nm):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=name,
        line_color='red', # Change the line color
        fillcolor='rgba(255, 0, 0, 0.1)', # Change the fill color
    ))
    fig.add_trace(go.Scatterpolar(
        r=user,
        theta=categories,
        fill='toself',
        name=user_nm,
        line_color='blue', # Change the line color
        fillcolor='rgba(0, 0, 255, 0.1)', # Change the fill color
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f'{comp_nm} - {name}'
    )

    st.plotly_chart(fig, use_container_width=True)
