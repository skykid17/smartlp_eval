import pcre2

regex = r"(?P<@timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\s+(?P<host>.*?)\s+(?P<process>.*?)\s+\[(?P<log.level>.*?)\]:\s+(?P<message>.*)"

log = "2025-03-07T11:37:22.109Z server52 DatabaseConnector [INFO]: Operation started for user 520"

def reduce(log: str, regex: str) -> str:
    # Ensure regex is a string
    if not isinstance(regex, str):
        print(f"Warning: Expected regex to be a string, got {type(regex)}. Converting to string.")
        regex = str(regex)
        
    compiled = None
    while regex:
        try:
            compiled = pcre2.compile(regex, pcre2.MULTILINE)
            if compiled.search(log):
                break
        except Exception:
            pass  # Skip invalid patterns silently
        regex = regex[:-1]
    return regex

print(reduce(log, regex))