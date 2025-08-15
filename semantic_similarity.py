from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def is_semantic_match(val1, val2):
    emb1 = model.encode(val1, convert_to_tensor=True)
    emb2 = model.encode(val2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

semantic_match = is_semantic_match("apple", "application")
print(f"Semantic match: {semantic_match}")