from sentence_transformers import SentenceTransformer

class SentenceTransformerModel:
    # We use a sentence transformers model, paraphrase-MiniLM-L6-v2 
    # Model size : 22.7M params
    def __init__(self, model_name='sentence-transformers/paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def forward(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings

