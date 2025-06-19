from flask import Flask, render_template, request
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text
import io
import base64
import numpy as np


app = Flask(__name__)

# Load and preprocess once at startup
def get_df_from_csv(filepath):
    df = pd.read_csv(filepath, index_col=0).dropna(how='all').dropna(axis=1, how='all')
    return df

def reduce_dim(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)
    result = pd.DataFrame(pca_data, columns=['PC1', 'PC2'], index=df.index)
    return result

df = get_df_from_csv('music_map_csv.csv')
pca_df = reduce_dim(df)

@app.route("/", methods=["GET"])
def scatter():
    highlight = request.args.get("highlight", "")  # optional highlight
    recommendations = []

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], label='All Artists')

    if highlight and highlight in pca_df.index:
        point = pca_df.loc[highlight]
        plt.scatter(point['PC1'], point['PC2'], color='yellow', s=100, label=f'{highlight} (Selected)')
        recommendations = get_recs(highlight, pca_df)
        #plt.legend()

    texts = []
    for i, label in enumerate(pca_df.index):
        x = pca_df.iloc[i, 0]
        y = pca_df.iloc[i, 1]
        texts.append(plt.text(x, y, label, fontsize=9))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.title("Jackson's Vibe-Based Music Map")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return render_template("map.html", plot=encoded, current_artist=highlight, recommendations=recommendations)

def get_recs(input, pca_df):
    try:
        point = pca_df.loc[input]
    except KeyError:
        return("That band/artist is not in my database. Please check spelling or try another artist.")

    copy = pca_df.copy()
    copy['distance'] = np.sqrt((copy['PC1'] - point.iloc[0])**2 + (copy['PC2'] - point.iloc[1])**2)

    # Sort by distance and exclude the target point itself if it's in the DataFrame
    nearest_points = copy[copy['distance'] != 0].nsmallest(3, 'distance')

    return nearest_points.index.tolist()

if __name__ == "__main__":
    app.run(debug=True)