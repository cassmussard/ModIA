import pandas as pd
from annoy import AnnoyIndex
from script_model.nlp_inference.nlp_model import SentenceTransformerModel

def preprocess(data_csv = 'movies_metadata.csv') : 
    model = SentenceTransformerModel()
    data = pd.read_csv(data_csv)
    data.dropna(subset=['original_title'], inplace=True)
    data.dropna(subset=['overview'], inplace=True)
    data = data[['original_title', 'overview']]
    return model, data


def create_annoy_vec(model, data, dim, name_file):
    dimension = dim
    n_trees = 10
    index = AnnoyIndex(dimension, 'angular')
    resumes = data['overview'].to_list()
    for i, resume in enumerate(resumes) :
        embedding = model.forward(resume)
        index.add_item(i, embedding)
    index.build(n_trees)
    index.save(name_file)

def recommend(model, data, query, k = 5):
    annoy_index = AnnoyIndex(384, 'angular')
    annoy_index.load('annoy_db_text.ann')
    query_emb = model.forward(query)
    indices = annoy_index.get_nns_by_vector(query_emb, k)
    # Top5 recommendation film title and their description
    top5_titles = [data.iloc[indice]['original_title'] for indice in indices]
    top5_overviews = [data.iloc[indice]['overview'] for indice in indices]
    return top5_titles, top5_overviews

if __name__ == '__main__':
    # Create Annoy index if not already done
    model, data = preprocess()
    create_annoy_vec(model, data, 384, 'annoy_db_text.ann')  # 384 is the dimension of the pooling layer of our model sentence-transformers/paraphrase-MiniLM-L6-v2
