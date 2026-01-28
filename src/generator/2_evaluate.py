import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from time import time

import pandas as pd
import pcre2
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Point to project root (3 levels up: file -> generator -> src -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.rag.mongo import rag_service

# Centralize file locations to match the new input/output layout
INPUT_DIR = BASE_DIR / "data" / "eval" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "eval" / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

lab_llm = ChatOpenAI(
    base_url="http://192.168.125.31:8000/v1",
    api_key="EMPTY",
    model="qwen25-coder-32b-awq",
    temperature=0.2
)

lab_llm_finetuned = ChatOpenAI(
    base_url="http://192.168.125.31:8001/v1",
    api_key="EMPTY",
    model="/home/rdpuser3/Downloads/qwen-2.5-coder-finetuned",
    temperature=0.2
)

SYSTEM_PROMPT = r'''You are a deterministic PCRE2 regex generator for log parsing.

Your ONLY task is to output a single PCRE2-compatible regular expression that parses the given log line.

## CRITICAL OUTPUT REQUIREMENTS
1. Output ONLY the regex pattern itself
2. No explanations, comments, markdown, code blocks, or extra text
3. Single line, no leading/trailing whitespace
4. No ^ anchor at start, no $ anchor at end (system adds $ automatically)

## SYNTAX REQUIREMENTS
5. Use PCRE2 syntax exclusively
6. All capture groups MUST be named using (?<name>...) syntax
7. Each capture group captures exactly ONE logical field
8. Capture ALL identifiable fields: timestamp, host, service, pid, ip, port, user, status, severity, rule_id, etc.
9. Free-text message bodies: capture as (?<content>...), NEVER use "message" as field name

## PATTERN CONSTRUCTION RULES
10. Use .*? ONLY inside capture groups, NEVER outside
11. Replace literal whitespace outside groups with \s+
12. Escape ALL regex metacharacters when matching literally: . [ ] ( ) { } + * ? ^ $ | \ /
13. Do NOT hard-code specific values that vary between logs (e.g., specific IPs, usernames)
14. Do NOT use overly-greedy quantifiers - prefer bounded or non-greedy versions
15. Prefer explicit character classes over wildcards when field format is known

## NAMED CAPTURE GROUP CONVENTIONS
16. Names must be: lowercase, underscore_separated, descriptive
17. Be consistent with field naming:
    - IP addresses: source_ip, dest_ip, client_ip, server_ip (not src/dst)
    - Ports: source_port, dest_port (not src_port)
    - Processes: process_name, process_id (not proc, pid alone)
    - Time: timestamp, time, datetime (not ts, date)
18. If provided with RAG examples showing field names, use those exact names for consistency

## HANDLING OPTIONAL FIELDS
19. To make a field optional, wrap the ENTIRE pattern (including delimiters) in non-capturing optional group
20. Correct: (?:\[(?<pid>\d+)\])? 
21. Incorrect: (?<pid>\d+)?  (never make named group itself optional)
22. Include surrounding delimiters/whitespace in the optional wrapper

## PATTERN SPECIFICITY GUIDELINES
23. Timestamps:
    - Syslog: (?<month>\w{3})\s+(?<day>\d{1,2})\s+(?<time>\d{2}:\d{2}:\d{2})
    - ISO8601: (?<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)
24. IP addresses: (?<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})
25. Ports: (?<port>\d{1,5})
26. PIDs: (?<pid>\d+)
27. HTTP status: (?<status>\d{3})
28. Quoted strings: "(?<field>[^"]*)"
29. Key=value: (?<key>\w+)=(?<value>\S+)
30. Hostnames: (?<host>[\w\.-]+)
31. Paths: (?<path>\S+) or (?<path>[^\s,\]]+) depending on delimiters

## FORMAT-SPECIFIC PATTERNS
32. Syslog: timestamp + host + process[pid]: + message
33. JSON logs: Look for key: value patterns, use [^"]+ for string values
34. CSV/Delimited: Identify delimiter, capture fields between them
35. Key=value pairs: Parse each as key=value pattern
36. Mixed format: Parse structured fields explicitly, capture remainder as (?<content>.+?)

## CHARACTER ESCAPING
37. Escape these characters when matching them literally: . [ ] ( ) { } + * ? ^ $ | \ /
38. Example: IP 192.168.1.1 → 192\.168\.1\.1
39. Example: Path /var/log → \/var\/log
40. Do NOT double-escape (no \\d, just \d) - system handles string encoding

## CONTEXT INTEGRATION (when RAG examples are provided)
41. Use field names and patterns from examples to maintain consistency
42. If multiple examples use a specific pattern for timestamps/IPs, follow that pattern
43. Prioritize field names that appear across multiple examples

## COMMON PATTERNS TO MEMORIZE
- Optional bracketed PID: (?:\[(?<pid>\d+)\])?
- Optional key=value: (?:key=(?<key_name>\S+))?
- IP with optional port: (?<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(?<port>\d{1,5}))?
- Quoted or unquoted value: (?:"(?<value>[^"]*)"|(?<value>\S+))
- Bracketed field: \[(?<field>[^\]]+)\]
- Free-text to end: (?<content>.+)

OUTPUT: Regex pattern only, nothing else.'''

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

def clean_msg(message):
    if not message:
        return ""
    message = str(message).strip()
    if message.startswith("```"):
        message = message[3:]
    if message.startswith("regex"):
        message = message[6:]
    if message.startswith("\n"):
        message = message[1:]
    if message.endswith("```"):
        message = message[:-3]
    return message.strip()

def reduce(log: str, regex: str) -> str:
    regex = str(regex)
    while regex:
        try:
            compiled = pcre2.compile(regex, pcre2.MULTILINE)
            if compiled.search(log):
                break
        except Exception:
            pass
        regex = regex[:-1]
    return regex

def resolve_duplicate_capture_groups(regex: str) -> str:
    """Identifies duplicate named capture groups and appends incremental numbers."""
    pattern = re.compile(r"(\(\?P?<)(\w+)(>)")
    seen = {}
    offset = 0

    for match in list(pattern.finditer(regex)):
        prefix, name, suffix = match.groups()
        start, end = match.span(2)

        if name in seen:
            seen[name] += 1
            new_name = f"{name}{seen[name]}"
            regex = regex[:start + offset] + new_name + regex[end + offset:]
            offset += len(new_name) - len(name)
        else:
            seen[name] = 1
    return regex

def escape_quotes(regex: str) -> str:
    """Escapes unescaped single and double quotes in the regex string."""
    escaped = []
    for i, ch in enumerate(regex):
        if ch in ["'", '"'] and (i == 0 or regex[i - 1] != '\\'):
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)

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
                if val_score > 0.7:
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
        return 0.0, {}

def generate_regex_direct(truth_df):
    """Generate regex using direct LLM calls."""
    results = []
    for idx, row in tqdm(truth_df.iterrows(), total=len(truth_df), desc="Generating (direct)"):
        log_id = str(row['log_id'])
        log = str(row['log_text'])
        messages = [("system", SYSTEM_PROMPT), ("human", log)]
        response = clean_msg(lab_llm.invoke(messages).content)
        results.append({"log_id": int(log_id), "generated_regex": response})
    return pd.DataFrame(results)

def generate_regex_finetuned(truth_df):
    """Generate regex using finetuned LLM."""
    results = []
    for idx, row in tqdm(truth_df.iterrows(), total=len(truth_df), desc="Generating (finetuned)"):
        log_id = str(row['log_id'])
        log = str(row['log_text'])
        messages = [("system", SYSTEM_PROMPT), ("human", log)]
        response = clean_msg(lab_llm_finetuned.invoke(messages).content)
        results.append({"log_id": int(log_id), "generated_regex": response})
    return pd.DataFrame(results)

def generate_regex_rag(truth_df):
    """Generate regex using RAG."""
    results = []
    for idx, row in tqdm(truth_df.iterrows(), total=len(truth_df), desc="Generating (RAG)"):
        log_id = str(row['log_id'])
        log = str(row['log_text'])
        rag_response = rag_service.query_rag(log, SYSTEM_PROMPT, filter_category="elastic_fields", verbose=False)
        
        if isinstance(rag_response, dict):
            response = clean_msg(rag_response.get("content", ""))
        elif isinstance(rag_response, (list, tuple)):
            response = clean_msg(rag_response[0] if rag_response else "")
        else:
            response = clean_msg(rag_response)
        
        results.append({"log_id": int(log_id), "generated_regex": response})
    return pd.DataFrame(results)

def generate_regex_decomposed_rag(truth_df):
    """Generate regex using decomposed RAG approach."""
    results = []
    query_count = 0
    
    for idx, row in tqdm(truth_df.iterrows(), total=len(truth_df), desc="Generating (decomposed RAG)"):
        log_id = str(row['log_id'])
        log_text = str(row['log_text'])
        
        regex = ""
        remaining_log = log_text
        attempt = 0
        max_attempts = 10

        while remaining_log.strip() and attempt < max_attempts:
            # Skip if remaining log is too short or only whitespace/punctuation
            if len(remaining_log.strip()) < 3:
                break
            
            print(f"attempt {attempt}")
            
            try:
                rag_response = rag_service.query_rag(remaining_log, SYSTEM_PROMPT, filter_category="elastic_fields", verbose=False)
            except Exception as e:
                print(f"RAG query failed for log_id {log_id}: {e}")
                break
            
            # Handle error responses
            if isinstance(rag_response, dict):
                if not rag_response.get("success", True):
                    print(f"RAG error for log_id {log_id}: {rag_response.get('error', 'Unknown error')}")
                    break
                response = clean_msg(rag_response.get("content", ""))
            elif isinstance(rag_response, (list, tuple)):
                response = clean_msg(rag_response[0] if rag_response else "")
            else:
                response = clean_msg(rag_response)
            
            if not response:
                break
            
            query_count += 1
            reduced_regex = reduce(remaining_log, response)

            try:
                compiled = pcre2.compile(reduced_regex, pcre2.MULTILINE)
                match = compiled.search(remaining_log)
            except Exception:
                break

            if not match:
                break

            remaining_log = remaining_log[match.end():].replace("\n", "")
            gap = log_text[match.end():match.end()+1]
            
            if regex and gap.isspace():
                regex += r"\s*" + reduced_regex
            else:
                regex += reduced_regex
            
            attempt += 1
            

        regex = escape_quotes(resolve_duplicate_capture_groups(regex))
        results.append({"log_id": int(log_id), "generated_regex": regex})
    
    return pd.DataFrame(results), query_count

def evaluate_results(df, truth_df, ground_truth_fields):
    """Evaluate generated regex patterns."""
    results = {}
    
    for i in tqdm(range(len(df)), desc="Evaluating"):
        log_id = str(df['log_id'].iloc[i])
        log_text = truth_df['log_text'].iloc[i]
        gt_regex = truth_df['ground_truth_regex'].iloc[i]
        gt_fields = ground_truth_fields[log_id]["extracted_fields"]
        regex = df['generated_regex'].iloc[i]
        
        em_acc = exact_match_accuracy(regex, gt_regex)
        func_acc = functional_accuracy(regex, log_text, gt_fields)
        precision, recall = field_level_precision_recall(regex, log_text, gt_fields)
        comp_ratio, extracted_fields = compilation_ratio(regex, log_text)
        
        results[log_id] = {
            "exact_match_accuracy": em_acc,
            "functional_accuracy": func_acc,
            "field_precision": precision,
            "field_recall": recall,
            "compilation_ratio": comp_ratio,
            "extracted_fields": extracted_fields
        }
    
    return results

def run_scenario(scenario):
    """Run complete generation and evaluation for a scenario."""
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario}")
    print(f"{'='*60}\n")
    
    # Load input data once
    truth_df = pd.read_csv(INPUT_DIR / "generator_golden_regex.csv", quoting=1)
    with open(INPUT_DIR / "generator_golden_fields.json", encoding="utf-8") as f:
        ground_truth_fields = json.load(f)
    
    # Generate regex (timed)
    start_time = time()
    query_count = 0
    
    if scenario == "direct":
        result_df = generate_regex_direct(truth_df)
    elif scenario == "finetuned":
        result_df = generate_regex_finetuned(truth_df)
    elif scenario == "rag":
        result_df = generate_regex_rag(truth_df)
    elif scenario == "decomposed_rag":
        result_df, query_count = generate_regex_decomposed_rag(truth_df)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    generation_time = time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    
    # Save generated regex
    output_file = OUTPUT_DIR / f"generator_output_{scenario}_2.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Saved regex to {output_file}")
    
    # Evaluate immediately
    print("\nEvaluating results...")
    results = evaluate_results(result_df, truth_df, ground_truth_fields)
    
    # Add timing info
    results["time_taken"] = generation_time
    results["query_count"] = query_count
    
    # Save evaluation results
    results_file = OUTPUT_DIR / f"generator_results_{scenario}_2.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")
    
    # Print summary
    metrics = ["exact_match_accuracy", "functional_accuracy", "field_precision", "field_recall", "compilation_ratio"]
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    
    for metric in metrics:
        values = [results[log_id][metric] for log_id in results if log_id not in ["time_taken", "query_count"]]
        avg = sum(values) / len(values) if values else 0
        print(f"{metric}: {avg:.4f}")
    
    print(f"\nTotal time: {generation_time:.2f}s")
    if query_count > 0:
        print(f"Total queries: {query_count}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    scenario = "decomposed_rag"  # Change this to switch scenarios: direct, finetuned, rag, decomposed_rag
    run_scenario(scenario)