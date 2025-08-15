import pcre2
import pandas as pd
from difflib import SequenceMatcher
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def string_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def semantic_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


SYNONYMS = {
    "app": "application",
    "src_ip": "source_ip",
    "dest_ip": "destination_ip",
    "msg": "message"
}

def normalize_value(value):
    return SYNONYMS.get(value.lower(), value.lower())

def exact_match_accuracy(generated_regex, ground_truth_regex):
    return string_similarity(generated_regex, ground_truth_regex)

def functional_accuracy(generated_regex, log_text, ground_truth_fields):
    try:
        pattern = pcre2.compile(generated_regex)
        match = pattern.match(log_text)
        if not match:
            return 0.0
        extracted = match.groupdict()
        print(extracted)
        correct = 0
        for gt_field, gt_value in ground_truth_fields.items():
            gt_norm_field = normalize_value(gt_field).lower()
            extracted_field_name = None
            # Tolerant field name matching using lower() and semantic similarity
            for field in extracted.keys():
                field_norm = field.lower()
                if field_norm == gt_norm_field or string_similarity(field_norm, gt_norm_field) > 0.8:
                    extracted_field_name = field
                    break
            if extracted_field_name and string_similarity(extracted[extracted_field_name], gt_value) > 0.7:
                correct += 1
        return correct / len(ground_truth_fields) if ground_truth_fields else 0
    except Exception as e:
        print(f"FA - Regex compilation error: {e}")
        return 0

def field_level_precision_recall(generated_regex, log_text, ground_truth_fields):
    try:
        pattern = pcre2.compile(generated_regex)
        match = pattern.match(log_text)
        if not match:
            return 0, 0
        extracted = match.groupdict()
        true_positive = 0
        for gt_field, gt_value in ground_truth_fields.items():
            gt_norm_field = normalize_value(gt_field).lower()
            matched_field = None
            for field in extracted.keys():
                field_norm = field.lower()
                if field_norm == gt_norm_field or string_similarity(field_norm, gt_norm_field) > 0.8:
                    matched_field = field
                    break
            if matched_field and string_similarity(str(extracted[matched_field]), str(gt_value)) > 0.9:
                true_positive += 1
        precision = true_positive / len(extracted) if extracted else 0
        recall = true_positive / len(ground_truth_fields) if ground_truth_fields else 0
        return precision, recall
    except Exception:
        return 0, 0

def compilation_success(generated_regex, log_text):
    try:
        pattern = pcre2.compile(generated_regex)
        match = pattern.search(log_text)
        print(match.group())
        return len(match.group())/len(log_text)
    except Exception:
        print("Compilation failed")
        return 0.0

with open("ground_truth_fields.json", encoding="utf-8") as f:
    ground_truth_fields = json.load(f)

df = pd.read_csv("rag_model.csv")
truth_df = pd.read_csv("ground_truth_regex.csv")
for i in range(11):
    print(10*"="+f"log {i}"+ 10*"=")
    log_id = str(df['log_id'].iloc[i])
    log_text = df['log_text'].iloc[i]
    gt_regex = truth_df['ground_truth_regex'].iloc[i]
    gt_fields = ground_truth_fields[log_id]["extracted_fields"]
    regex = df['generated_regex'].iloc[i]
    print(f"log is {log_text}\n")
    print(f"regex is {regex}\n")
    print(f"gt_regex is {gt_regex}\n")


    # ==== Evaluation ====

    em_acc = exact_match_accuracy(regex, gt_regex)
    func_acc = functional_accuracy(regex, log_text, gt_fields)
    precision, recall = field_level_precision_recall(regex, log_text, gt_fields)
    comp = compilation_success(regex, log_text)
    print(f"Exact Match Accuracy is {em_acc}")
    print(f"Functional Accuracy is {func_acc}")
    print(f"Field Precision is {precision}")
    print(f"Field Recall is {recall}")
    print(f"Compilation Ratio is {comp}")
