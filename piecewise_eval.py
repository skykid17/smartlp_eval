import re
import pandas as pd
from difflib import SequenceMatcher
import json

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

SYNONYMS = {
    "app": "application",
    "src_ip": "source_ip",
    "dest_ip": "destination_ip",
    "msg": "message"
}

def normalize_value(value):
    return SYNONYMS.get(value.lower(), value.lower())

def exact_match_accuracy(generated_regex, ground_truth_regex):
    return 1.0 if generated_regex == ground_truth_regex else string_similarity(generated_regex, ground_truth_regex)

def functional_accuracy(generated_regex, log_text, ground_truth_fields):
    try:
        pattern = re.compile(generated_regex)
        match = pattern.match(log_text)
        if not match:
            return 0.0
        extracted = match.groupdict()
        
        correct = 0
        for gt_field, gt_value in ground_truth_fields.items():
            # normalize aliases
            norm_field = normalize_value(gt_field)

            # check if extracted has a matching field name (original or normalized)
            extracted_field_name = None
            for f in extracted.keys():
                if f == gt_field or f == norm_field:
                    extracted_field_name = f
                    break

            # compare the **values**, not field names
            if extracted_field_name and extracted[extracted_field_name] == gt_value:
                correct += 1

        return correct / len(ground_truth_fields) if ground_truth_fields else 0
    except re.error:
        return 0.0
    except re.error as e:
        print(f"FA - Regex compilation error: {e}")
        return 0

def field_level_precision_recall(generated_regex, log_text, ground_truth_fields):
    try:
        pattern = re.compile(generated_regex)
        match = pattern.match(log_text)
        if not match:
            return 0.0, 0.0
        extracted = match.groupdict()
        correct_fields = set(extracted.keys()) & set(ground_truth_fields.keys())
        true_positive = 0
        for field in correct_fields:
            sim = string_similarity(str(extracted[field]), str(ground_truth_fields[field]))
            if sim > 0.9:  # high similarity threshold
                true_positive += 1
        precision = true_positive / len(extracted) if extracted else 0.0
        recall = true_positive / len(ground_truth_fields) if ground_truth_fields else 0.0
        return precision, recall
    except re.error:
        return 0.0, 0.0

def compilation_success(generated_regex):
    try:
        re.compile(generated_regex)
        return 1.0
    except re.error:
        return 0.0

with open("ground_truth_fields.json", encoding="utf-8") as f:
    ground_truth_fields = json.load(f)

df = pd.read_csv("rag_model.csv")
truth_df = pd.read_csv("ground_truth_regex.csv")
log_id = str(df['log_id'].iloc[0])
log_text = df['log_text'].iloc[0]
gt_regex = truth_df['ground_truth_regex'].iloc[0]
gt_fields = ground_truth_fields[log_id]["extracted_fields"]
regex = df['generated_regex'].iloc[0]
print(f"log is {log_text}")
print(f"regex is {regex}")
print(f"gt_regex is {gt_regex}")

# ==== Evaluation ====

em_acc = exact_match_accuracy(regex, gt_regex)
func_acc = functional_accuracy(regex, log_text, gt_fields)
precision, recall = field_level_precision_recall(regex, log_text, gt_fields)
comp_rate = compilation_success(regex)
print(f"Exact Match Accuracy is {em_acc}")
print(f"Functional Accuracy is {func_acc}")
print(f"Field Precision is {precision}")
print(f"Field Recall is {recall}")
if comp_rate:
    print(f"Compilation Success")
else:
    print(f"Compilation Failed")
