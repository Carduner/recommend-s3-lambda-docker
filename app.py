from s3fs.core import S3FileSystem
import spotipy
import pandas as pd
import os
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")




def audio_features(ids,sp):
    '''takes in track ids and return df with all features
        -- audio features
        -- related genres
    '''
    # get audio features for songs
    audio_features = sp.audio_features(ids)
    audio_df = pd.DataFrame(audio_features)
    
    # merge into my_tracks
    audio_cols = ['id','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    return audio_df[audio_cols]

def genre_transformer(artist_ids,bucket_name,transfomer_pipeline_path):
    '''transforms genre df into principal components using svd
        --
    
    '''
    genre_results = {} # initiate container for results
    for artist_id in artist_ids:
        genre_set = set() # use set so we don't dupe genres
        try:
            artist = sp.artist(artist_id) # get artist genre
            genre_set.update(artist['genres'])
        except:
            continue
        try:
            df_related_artists = pd.DataFrame(sp.artist_related_artists(artist_id)['artists']) # get related artist
            # update set with all genres from related artists
            for i in df_related_artists.genres:
                genre_set.update(i)
        except:
            continue
        genre_results[artist_id] = list(genre_set)
    
    genres = pd.Series(genre_results).to_frame('genres')
    
    # load transformers
    s3_file = S3FileSystem()
    transformers = pickle.load(s3_file.open('{}/{}'.format(bucket_name, transfomer_pipeline_path)))
    mlb = transformers['mlb']
    svd = transformers['svd']
    genres_oh = pd.DataFrame(mlb.transform(genres.genres),
                         columns=mlb.classes_,
                         index=genres.index)
    genre_svd_df = pd.DataFrame(svd.transform(genres_oh), 
                                 index = genres_oh.index)
    genre_svd_df = genre_svd_df.reset_index()
    genre_svd_df = genre_svd_df.rename(columns={'index':'artist_id'})
    
    return genre_svd_df
    
def similarity(X,Y,df_fs):
    ''' takes in transformed df and computes similarity with feature store, returns recommended track ids
    '''
    sim_matrix = cosine_similarity(X,Y)
    df_sim = pd.DataFrame(sim_matrix, index=df_fs.index)
    df_sim['sum'] = df_sim.sum(axis=1)
    df_sim[['id','artist_id','name','popularity']] = df_fs[['id','artist_id','name','popularity']]
    df_recs = df_sim[['id','artist_id','name','popularity','sum']].sort_values(by=['sum'],ascending=False).head(50)
    
    return df_recs

def dedupe(df):
    df['name_artist'] = df['name'] + df['artist_id']
    idx = df.groupby(['name_artist'])['popularity'].transform(max) == df['popularity']
    return df[idx].groupby(['name_artist']).first().reset_index().drop(columns=['name_artist'])

def main(sp):
    bucket_name = 'recommend-spotify'
    base_songs_path = 'base_songs_landing/tracks.json'
    data_location = 's3://{}/{}'.format(bucket_name, base_songs_path)

    df = pd.read_json(data_location)

    track_ids = df.id.tolist()
    artist_ids = df.artist_id.tolist()
    transformer_pipeline_path = 'sklearn_objects/objects_20220803143707.pkl'
    
    audio_df = audio_features(track_ids,sp)
    genre_pca_df = genre_transformer(artist_ids, bucket_name, transformer_pipeline_path)
    
    # merge audio_df and genre_pca_df on artist_id left join
    df = df.merge(audio_df,how='left',on='id')
    df = df.merge(genre_pca_df,how='left',on='artist_id')
    replace_col_names = {x:'{}'.format(x) for x in range(0,100)}
    df = df.rename(columns=replace_col_names)
    
    #compute similarity and write rec_ids to out file
    fs_path = 'feature_store/tracks_fs_20220811142647.csv'
    data_location = 's3://{}/{}'.format(bucket_name, fs_path)
    df_fs = pd.read_csv(data_location,index_col=0)
    df_fs = df_fs.fillna(0)
    id_cols = ['id','artist_id','name','popularity']
    feature_cols = [x for x in df_fs.columns.values.tolist() if x not in id_cols]
    X = df_fs[feature_cols]
    Y = df[feature_cols]
    df_recs = similarity(X,Y,df_fs)
    df_recs = dedupe(df_recs)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    fn = 'rec_ids_{}.json'.format(ts)
    out_directory = 'recommended_tracks'
    s3_path = 's3://{}/{}/{}'.format(bucket_name, out_directory, fn)
    s3 = S3FileSystem(anon=False)
    with s3.open(s3_path,'w') as f:
        df_recs.to_json(f, orient='records')

def handler(event, context):
    
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ.get("SPOTIPY_CLIENT_ID"),
                                                                    client_secret=os.environ.get("SPOTIPY_CLIENT_SECRET")))
        main(sp)
    
        return {
            'statusCode': 200,
            'message':'Success!'}
    except:
        return {
            'statusCode': 400,
            'body': 'Error, bad request!'}

