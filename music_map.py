import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np

def get_df_from_csv(filepath):
    #read in csv as dataframe
    df = pd.read_csv(filepath, index_col=0)
    #clean data
    
    # Drop rows where all cells are empty
    df = df.dropna(how='all')
    # drop columns where all cells are empty
    df = df.dropna(axis=1, how='all')

    return df   

def reduce_dim(df):
    #scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    #reduce to two components
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2']) 
    pca_df.index = df.index.tolist()

    # return a DataFrame with the results
    return pca_df


def plot_map(df, pca_df):

    #set size
    plt.figure(figsize=(8,6))

    # make scatter plot
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    map_input = input("Type in the name of an artist you would like to locate on the map. If you just want to see the plain map, hit enter. ")
    if map_input != "":
        try:
            row = pca_df.loc[map_input]
            plt.scatter(row.loc['PC1'], row.loc['PC2'], color='yellow')
        except KeyError:
            print("That band/artist is not in my database. Here is the plain map.")

    # Add labels for each point and title
    plt.title('Jackson''s Vibe-Based Music Map')
    texts = []
    for i, label in enumerate(df.index.tolist()):
        x = pca_df['PC1'].iloc[i]
        y = pca_df['PC2'].iloc[i]
        texts.append(plt.text(x, y, label, fontsize=9))

    #adjust overlapping text
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    #display
    plt.grid(True)
    plt.show()

def get_recs(input, pca_df):
    try:
        point = pca_df.loc[input]
    except KeyError:
        print("That band/artist is not in my database. Please check spelling or try another artist.")
        return

    print('You like "' + input + '". Based off this preference, I think you might also like the following artists/bands: ')
    copy = pca_df.copy()
    copy['distance'] = np.sqrt((copy['PC1'] - point.iloc[0])**2 + (copy['PC2'] - point.iloc[1])**2)

    # Sort by distance and exclude the target point itself if it's in the DataFrame
    nearest_points = copy[copy['distance'] != 0].nsmallest(3, 'distance')

    print(nearest_points.index.tolist())

df = get_df_from_csv('music_map_csv.csv')
pca_df = reduce_dim(df)
user_option = input('Hello. Type "map" if you would you like to view music map, or type "rec" to get a band recommendation. ')
while (user_option !='q'):
    if user_option == "map":
        plot_map(df, pca_df)
    elif user_option == "rec":
        user_band = input('Type in a band or musician that you like. ')
        get_recs(user_band, pca_df)
    user_option = input('Type "map" if you would you like to view music map, or type "rec" to get a band recommendation. Press q to quit. ')





