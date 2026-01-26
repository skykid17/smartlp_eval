
"""Generate `golden_documents.json` by finding files with certain
extensions, filtering out any path that contains the string "license",
checkpointing the filtered list, then selecting 100 random paths and
prefixing each with `data/`.

Saves two files in the current directory:
- `filtered_paths_checkpoint.json` : JSON array of all filtered file paths
  (pre-sampling, paths are relative and POSIX-style)
- `golden_documents.json` : JSON array of 100 (or fewer) file paths,
  each prefixed with `data/` as requested.

Run: `python get_golden_documents.py`
"""

from pathlib import Path, PurePosixPath
import json
import random
import sys
import os
import logging
from typing import Any, List

MIN_DOC_LENGTH = 50
EXTENSIONS = {'.txt', '.docx', '.pdf', '.md', '.markdown', '.mdx'}

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def gather_paths(root: Path) -> list:
	results = []
	for p in root.rglob('*'):
		if p.is_file():
			if p.suffix.lower() in EXTENSIONS:
				# Use POSIX-style path (forward slashes) and relative to root
				try:
					rel = p.relative_to(root)
				except ValueError:
					rel = p
				posix = PurePosixPath(rel).as_posix()
				try:
					size = os.path.getsize(p)
				except OSError as e:
					print(f"Warning: Could not get size for file {p}: {e}", file=sys.stderr)
					continue
				if size == 0:
					continue
				results.append(posix)
	return results


def safe_load_json_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            f.seek(0)
            return [line.strip() for line in f.readlines() if line.strip()]

def filter_documents(paths_file: str) -> List[str]:
    paths = safe_load_json_lines(paths_file)
    valid_paths, skipped = [], 0
    for p in paths:
        if not os.path.exists(p) or any(x in p.lower() for x in ("cacert", "license")):
            skipped += 1
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if len(f.read().strip()) >= MIN_DOC_LENGTH:
                    valid_paths.append(p)
                else:
                    skipped += 1
        except:
            skipped += 1
    write_json(paths_file, valid_paths)
    logging.info(f"Filtered documents: kept {len(valid_paths)}, skipped {skipped}")
    return valid_paths

def main():
	root = Path('data/')
	print('Scanning for files...')
	all_matches = gather_paths(root)
	print(f'Found {len(all_matches)} candidate files with target extensions.')

	filtered = filter_documents(all_matches)
	print(f'{len(filtered)} files remain after filtering out "license".')
	
	# Save checkpoint (pre-sampling)
	checkpoint_file = Path('data/eval/input/all_document_paths.json')
	data = [f'data/{p}' for p in filtered]
	with checkpoint_file.open('w', encoding='utf-8') as f:
		json.dump(data, f, indent=2, ensure_ascii=False)
	print(f'Wrote checkpoint JSON to {checkpoint_file!s}')

if __name__ == '__main__':
	main()

