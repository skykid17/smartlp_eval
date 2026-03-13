
# from rag_mongo import extract_keywords, LocalRetriever, rag_service
# from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer
import json
import re
from langchain_openai import ChatOpenAI
from tqdm import tqdm

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