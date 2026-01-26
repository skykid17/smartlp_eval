
# from rag_mongo import extract_keywords, LocalRetriever, rag_service
# from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer
import json
import re
from langchain_openai import ChatOpenAI
from tqdm import tqdm


# def build_local_retriever() -> LocalRetriever:
#     return LocalRetriever(
#         collection=rag_service._ensure_collection(),
#         embedding_fn=lambda texts, show_progress=False: rag_service.generate_embeddings(
#             texts, show_progress
#         ),
#         text_index=rag_service.text_index,
#         vector_index=rag_service.vector_index,
#     )


# def retrieve_vector_only(query: str, k: int = 5) -> list[Document]:
#     retriever = build_local_retriever()
#     sem_run = retriever.run_vector_search(
#         query=query,
#         semantic_k=k,
#         vector_index="vector_index"
#     )

#     return [
#         Document(
#             page_content=d["content"],
#             metadata={
#                 **d["metadata"],
#                 "doc_id": d["_id"],
#                 "score": d["score"],
#                 "retriever": "vector"
#             }
#         )
#         for d in sem_run
#     ]

# def retrieve_bm25_only(query: str, k: int = 5) -> list[Document]:
#     kw = extract_keywords(query, max_keywords=15)
#     retriever = build_local_retriever()
#     kw_run = retriever.run_text_search(
#         keyword_query=kw,
#         keyword_k=k,
#         base_filter={}
#     )

#     return [
#         Document(
#             page_content=d["content"],
#             metadata={
#                 **d["metadata"],
#                 "doc_id": d["_id"],
#                 "score": d["score"],
#                 "retriever": "bm25"
#             }
#         )
#         for d in kw_run
#     ]

# def retrieve_hybrid(query: str, k: int = 5) -> list[Document]:
#     retriever = build_local_retriever()

#     docs = retriever.run_hybrid_search(
#         query=query,
#         limit=k,
#     )

#     for d in docs:
#         d.metadata["retriever"] = "hybrid"

#     return docs


# with open("data/eval/input/retriever_golden_dataset.json", "r", encoding="utf-8") as f:
#         golden_dataset = json.load(f)

# print("Text Retrieval Evaluation")
# for items in golden_dataset:
#     items["retrieval_context"] = []
#     retreived_docs = retrieve_bm25_only(items["input"],k=5)
#     for doc in retreived_docs:
#         items["retrieval_context"].append(doc.page_content)

# with open(f"data/eval/output/processed_text.json", "w", encoding="utf-8") as f:
#     json.dump(golden_dataset, f, indent=2, ensure_ascii=False)
    
# print("Vector Retrieval Evaluation")
# for items in golden_dataset:
#     items["retrieval_context"] = []

#     retreived_docs = retrieve_vector_only(items["input"],k=5)
#     for doc in retreived_docs:
#         items["retrieval_context"].append(doc.page_content)

# with open(f"data/eval/output/processed_vector.json", "w", encoding="utf-8") as f:
#     json.dump(golden_dataset, f, indent=2, ensure_ascii=False)

# print("Hybrid Retrieval Evaluation")
# for items in golden_dataset:
#     items["retrieval_context"] = []
#     retreived_docs = retrieve_hybrid(items["input"],k=5)
#     for doc in retreived_docs:
#         items["retrieval_context"].append(doc.page_content)
# with open(f"data/eval/output/processed_hybrid.json", "w", encoding="utf-8") as f:
#     json.dump(golden_dataset, f, indent=2, ensure_ascii=False)

modes = ["hybrid", "text", "vector"]
llm = ChatOpenAI(
    model="Qwen3-8B",
    base_url="http://192.168.125.31:8000/v1",
    api_key="test",
    temperature=0,
)

for mode in modes:
    with open(f"data/eval/output/processed_{mode}.json", "r", encoding="utf-8") as f:
        retrieved_output = json.load(f)
    
    for item in tqdm(retrieved_output, desc=f"Generating answers for mode: {mode}"):
        context = item["retrieval_context"]
        prompt = f"""You are an AI assistant that helps people find information. Keep your responses concise and short. Use the following pieces of context to answer the question at the end. 
{''.join([f'Context {i+1}: {c}\n' for i, c in enumerate(context)])}
Question: {item['input']}"""
        response = llm.invoke(prompt)
        raw_content = response.content
        clean_content = re.sub(r"<think>.*?</think>", "", raw_content.strip(), flags=re.DOTALL).strip()
        item["actual_output"] = clean_content
        with open(f"data/eval/output/output_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(retrieved_output, f, indent=2, ensure_ascii=False)