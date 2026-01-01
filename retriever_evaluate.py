from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm

vllm_model = ChatOpenAI(
    model="qwen25-coder-32b-awq",
    base_url="http://192.168.125.31:8000/v1",
    api_key="test",
    model_kwargs={"response_format": {"type": "json_object"}},
)

# ----------------------------------------------------------
# INTERNAL: helper LLM judge function
# ----------------------------------------------------------

def llm_judge(prompt: str) -> dict:
    """Call vLLM using a JSON response format and return the parsed JSON."""
    resp = vllm_model.invoke(prompt)
    content = resp.content
    if isinstance(content, list):
        content = content[0].get("text", content[0])
    return json.loads(content)


# ----------------------------------------------------------
# METRIC 1 — Contextual Precision
# ----------------------------------------------------------

def contextual_precision(test_case, retrieved_docs):
    """
    Precision = (# of docs relevant to the question) / (# retrieved docs)
    """
    question = test_case["input"]
    total = len(retrieved_docs)
    if total == 0:
        return 0.0, "No retrieved documents"

    relevant_count = 0
    reasons = []

    for doc in retrieved_docs:
        prompt = f"""
You are evaluating contextual precision.

Question:
{question}

Retrieved Chunk:
{doc}

Task:
Determine if this retrieved chunk is relevant to answering the question.

Respond in JSON:
{{
  "relevant": true/false,
  "reason": "<short reason>"
}}
"""
        result = llm_judge(prompt)
        if result.get("relevant", False):
            relevant_count += 1
        reasons.append(result.get("reason", ""))

    precision = relevant_count / total
    reason_summary = f"{relevant_count}/{total} chunks relevant. Details: " + " | ".join(reasons)
    return precision, reason_summary


# ----------------------------------------------------------
# METRIC 2 — Contextual Recall
# ----------------------------------------------------------

def contextual_recall(test_case, retrieved_docs):
    """
    Recall = How much of the golden answer is supported by retrieved docs.
    Requires LLM to judge support level (0–1).
    """
    question = test_case["input"]
    golden_answer = test_case["expected_output"]
    context = "\n\n---\n\n".join(retrieved_docs)

    if not retrieved_docs:
        return 0.0, "No retrieved context, so no golden answer can be supported."

    prompt = f"""
You are evaluating contextual recall.

Question:
{question}

Golden Answer:
{golden_answer}

Retrieved Context:
{context}

Task:
Determine how much of the golden answer is supported by the retrieved context.
Score 0–1.

Meaning of score:
- 1 = All information in golden answer is fully supported.
- 0 = None of the golden answer is supported.
- 0.5 = About half of the information is supported.

Respond in JSON:
{{
  "score": <float 0-1>,
  "reason": "<short explanation>"
}}
"""
    result = llm_judge(prompt)
    score = float(result.get("score", 0))
    return score, result.get("reason", "")


# ----------------------------------------------------------
# METRIC 3 — Contextual Relevancy
# ----------------------------------------------------------

def contextual_relevancy(test_case, retrieved_docs):
    """
    Relevancy = 1 - (# irrelevant / total)
    Similar to precision but considers irrelevant noise penalties.
    """
    question = test_case["input"]
    total = len(retrieved_docs)
    if total == 0:
        return 0.0, "No retrieved docs."

    context = "\n\n---\n\n".join(retrieved_docs)

    prompt = f"""
You are evaluating contextual relevancy.

Question:
{question}

Retrieved Context:
{context}

Task:
Identify which retrieved chunks are irrelevant to the question.

Respond in JSON:
{{
  "irrelevant_indices": [list of indices 0..n-1],
  "reason": "<short explanation>"
}}
"""
    result = llm_judge(prompt)
    irrelevant = result.get("irrelevant_indices", [])
    irr_count = len(irrelevant)

    score = 1 - (irr_count / total)
    return score, result.get("reason", "")


# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------

def main() -> None:
    # modes = ["hybrid", "text", "vector"]
    modes = ["text", "vector"]
    for mode in modes:
        print(f"Evaluating mode: {mode}")
        with open(f"data/eval/output/retriever_output_{mode}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results = []
        for case in tqdm(dataset):
            retrieved_docs = case.get("retrieval_context", [])

            precision, prec_reason = contextual_precision(case, retrieved_docs)
            recall, rec_reason = contextual_recall(case, retrieved_docs)
            relevancy, rel_reason = contextual_relevancy(case, retrieved_docs)

            results.append({
                "input": case["input"],
                "contextual_precision": {
                    "score": precision,
                    "reason": prec_reason
                },
                "contextual_recall": {
                    "score": recall,
                    "reason": rec_reason
                },
                "contextual_relevancy": {
                    "score": relevancy,
                    "reason": rel_reason
                },
            })

        out_path = f"data/eval/output/retriever_results_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()