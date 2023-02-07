import numpy as np
from sentence_transformers import SentenceTransformer
import scipy
import time

def main():
    
    queries=input()
    
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    query_embeddings = model.encode(queries)
    sentence_embeddings=np.load(('./sentence_embeddings.npy'),allow_pickle=True)
    print("Search Results")
    number_top_matches = 5
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        print("Query:", query)
        print("\nTop {} most similar sentences in corpus:".format(number_top_matches))

        for idx, distance in results[0:number_top_matches]:
            print(dataset[idx])
    

if __name__ == "__main__":
    main()