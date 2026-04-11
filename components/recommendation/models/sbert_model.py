from sentence_transformers import SentenceTransformer

model = None

def get_model():
    global model
    if model is None:
        print("Loading SBERT model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return model