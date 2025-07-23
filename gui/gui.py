from flask import Flask, render_template, request, jsonify
import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from anime_recommendation_system import recommend_anime
from anime_recommendation_system import collaborative_filtering
from anime_recommendation_system import hybrid_recommend_for_user
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('anime.html')
@app.route('/recommend',  methods=["GET"])
def anime():
    title = request.args.get('title')
    if( title  is None):
          return  render_template('anime.html')
    return render_template('contect_base.html')
@app.route('/collaborative_filtering',  methods=["GET"])
def user():
    user_name = request.args.get('User_name')
    if(user_name is None):
          return  render_template('user.html')
    return render_template('collaborative_filtering.html')
@app.route('/userjs',  methods=["GET"])
def userjs():
    animeDf = pd.read_csv(r'data/anime_genre.csv')
    user_name = request.args.get('User_name')
    if(user_name is None):
          return  render_template('user.html')
    results = collaborative_filtering(user_name,10000)
    results =pd.merge(results, animeDf, on='title', how='inner')[['title', 'anime_year', 'score', 'img_url']]
    results=results.to_dict(orient='records')
    for item in results:
        for key, value in item.items():
            if isinstance(value, float) and pd.isna(value):
                item[key] = None
        if pd.isna(item.get('anime_year')):
            item['anime_year'] = None
        else:
            item['anime_year'] = int(item['anime_year'])
    return jsonify(results)
@app.route('/anime_profile', methods=["get"])
def anime_profile():
    animeDf = pd.read_csv(r'data/anime_genre.csv')
    title = request.args.get('title') 
    row=animeDf[animeDf['title']==title]
    reviewsDf = pd.read_csv("data/user_rate.csv")
    score_labels = ['Story_score', 'Animation_score', 'Sound_score', 'Character_score', 'Enjoyment_score']
    for col in score_labels:
        reviewsDf[col] = pd.to_numeric(reviewsDf[col], errors='coerce')

    anime_avg = reviewsDf.groupby("title")[score_labels].mean()
    anime_name = title
    if title not in anime_avg.index:
        return render_template('anime_profile.html'
                           ,title=row.iloc[0][1],Synopsis=row.iloc[0][2],Genre=row.iloc[0][3],
                           Aired=row.iloc[0][4],Episodes=row.iloc[0][5],Popularity=row.iloc[0][7]
                           ,Ranked=row.iloc[0][8],score=row.iloc[0][9],src=row.iloc[0][10])
    scores = anime_avg.loc[anime_name].tolist()
    labels = [label.replace("_score", "") for label in score_labels]
    scores += scores[:1]
    labels += labels[:1]

    # Radar chart with dark theme
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=labels,
        fill='toself',
        name=anime_name,
        line=dict(color='#ffd700'),
        fillcolor='rgba(255, 215, 0, 0.3)',
        marker=dict(color='#ffd700')
    ))

    fig.update_layout(
        title={
            'text': f"",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=22, color='#9896f1')
        },
        paper_bgcolor='rgba(7, 15, 43,0)',
        plot_bgcolor='rgba(27, 26, 85, 1)',
        font=dict(color='#fff', family='Poppins'),
        polar=dict(
            bgcolor='rgba(27, 26, 85, 1)',
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[0, 2, 4, 6, 8, 10],
                tickfont=dict(color='#f0f0f0'),
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(color='#ffd700'),
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            )
        ),
        showlegend=False
    )
    graph_html = pio.to_html(fig, full_html=False)
    return render_template('anime_profile.html'
                           ,title=row.iloc[0][1],Synopsis=row.iloc[0][2],Genre=row.iloc[0][3],
                           Aired=row.iloc[0][4],Episodes=row.iloc[0][5],Popularity=row.iloc[0][7]
                           ,Ranked=np.int64(row.iloc[0][8]),score=row.iloc[0][9],src=row.iloc[0][10], graph_html=  graph_html)
@app.route('/anime_profilejs')
def anime_profilejs():
    title = request.args.get('title')
    results = recommend_anime(title, 10000)
    for item in results:
        for key, value in item.items():
            if isinstance(value, float) and pd.isna(value):
                item[key] = None
        if pd.isna(item.get('anime_year')):
            item['anime_year'] = None
        else:
            item['anime_year'] = int(item['anime_year'])

    return jsonify(results)

@app.route('/hybrid',  methods=["GET"])
def hybrid():
    user_name = request.args.get('User_name')
    if(user_name is None):
          return  render_template('userh.html')
    return render_template('hybrid.html')
@app.route('/hybridjs',  methods=["GET"])
def hybridjs():
    animeDf = pd.read_csv(r'data/anime_genre.csv')
    user_name = request.args.get('User_name')
    if(user_name is None):
          return  render_template('userh.html')
    results = hybrid_recommend_for_user( user_name, k=10000, alpha=0.5)
    results =pd.merge(results, animeDf, on='title', how='inner')[['title', 'anime_year', 'score', 'img_url']]
    results=results.to_dict(orient='records')
    for item in results:
        for key, value in item.items():
            if isinstance(value, float) and pd.isna(value):
                item[key] = None
        if pd.isna(item.get('anime_year')):
            item['anime_year'] = None
        else:
            item['anime_year'] = int(item['anime_year'])
    return jsonify(results)

if __name__ == '__main__':
    app.run()
