import pcre2
import pandas as pd
from difflib import SequenceMatcher
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR = Path(__file__).resolve().parent
# Centralize file locations to match the new input/output layout
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def string_similarity(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def semantic_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


SYNONYMS = {
    "app": "application",
    "src_ip": "source_ip",
    "dest_ip": "destination_ip",
    "msg": "message",
    "pri": "priority",
    "pid": "process_id",
    "user_id": "uid",
    "log_level": "loglevel"
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
        correct = 0

        for gt_field, gt_value in ground_truth_fields.items():
            gt_norm_field = normalize_value(gt_field).lower()
            extracted_field_name = None

            # Match fields using semantic OR string similarity
            best_score = 0.0
            for field in extracted.keys():
                field_norm = field.lower()
                score = max(
                    string_similarity(field_norm, gt_norm_field),
                    semantic_similarity(field_norm, gt_norm_field)
                )
                if score > best_score:
                    best_score = score
                    extracted_field_name = field

            # Check if best match is strong enough
            if extracted_field_name and best_score > 0.75:
                extracted_val = str(extracted[extracted_field_name])
                val_score = max(
                    string_similarity(extracted_val, str(gt_value)),
                    semantic_similarity(extracted_val, str(gt_value))
                )
                if val_score > 0.7:  # threshold for values
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

def compilation_ratio(generated_regex, log_text):
    try:
        pattern = pcre2.compile(generated_regex)
        match = pattern.search(log_text)
        return len(match.group())/len(log_text), match.groupdict()
    except Exception:
        print("Compilation failed")
        return 0.0, {}

with open(INPUT_DIR / "ground_truth_fields.json", encoding="utf-8") as f:
    ground_truth_fields = json.load(f)


scenario = "rag"  # Change this to switch scenarios
df = pd.read_csv(OUTPUT_DIR / f"regex_{scenario}.csv")
truth_df = pd.read_csv(INPUT_DIR / "ground_truth_regex.csv", quoting=1)
results = {}

with open(OUTPUT_DIR / f"results_{scenario}.json", "r") as f:
    data = json.load(f)

for i in range(100):
    print(10*"="+f"log {i+1}"+ 10*"=")
    log_id = str(df['log_id'].iloc[i])
    log_text = truth_df['log_text'].iloc[i]
    gt_regex = truth_df['ground_truth_regex'].iloc[i]
    gt_fields = ground_truth_fields[log_id]["extracted_fields"]
    regex = df['generated_regex'].iloc[i]
    em_acc = exact_match_accuracy(regex, gt_regex)
    func_acc = functional_accuracy(regex, log_text, gt_fields)
    precision, recall = field_level_precision_recall(regex, log_text, gt_fields)
    comp_ratio, extracted_fields = compilation_ratio(regex, log_text)
    print(f"Exact Match Accuracy is {em_acc}")
    print(f"Functional Accuracy is {func_acc}")
    print(f"Field Precision is {precision}")
    print(f"Field Recall is {recall}")
    print(f"Compilation Ratio is {comp_ratio}")

    data[log_id] = {
            "exact_match_accuracy": em_acc,
            "functional_accuracy": func_acc,
            "field_precision": precision,
            "field_recall": recall,
            "compilation_ratio": comp_ratio,
            "extracted_fields": extracted_fields
        }
    # Save results to json

with open(OUTPUT_DIR / f"results_{scenario}.json", "w") as f:
    json.dump(data, f, indent=2)