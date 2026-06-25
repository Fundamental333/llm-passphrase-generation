from pathlib import Path
from collections import Counter, defaultdict
import gzip
import json
import pandas as pd


ROOT = Path("/content/drive/MyDrive/LLM_Passphrase_MaxCoverage_2M")
CHUNK_DIR = ROOT / "t4_llm_chunks_50k_v4" / "chunks"

valid_files = sorted(CHUNK_DIR.glob("chunk_*_valid_subset.jsonl.gz"))

print("Valid chunk files found:")
for fp in valid_files:
    print(fp.name, round(fp.stat().st_size / (1024 * 1024), 2), "MB")

if not valid_files:
    raise FileNotFoundError("No valid chunk files found.")


seen = set()
duplicate_counter = Counter()
family_counter = Counter()
family_unique_sets = defaultdict(set)
word_count_counter = Counter()
char_count_counter = Counter()
file_counter = Counter()

total_rows = 0
usable_rows = 0
bad_json = 0
empty_phrase = 0


def get_phrase(row):
    for key in ["cleaned_passphrase", "passphrase", "cleaned", "text", "output", "raw_output"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def get_family(row):
    for key in ["prompt_family", "family", "prompt_type", "category", "generator_type"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown"


for fp in valid_files:
    print("Reading:", fp.name)

    with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_rows += 1
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            phrase = get_phrase(row)

            if not phrase:
                empty_phrase += 1
                continue

            usable_rows += 1
            family = get_family(row)

            family_counter[family] += 1
            family_unique_sets[family].add(phrase)
            word_count_counter[len(phrase.split())] += 1
            char_count_counter[len(phrase)] += 1
            file_counter[fp.name] += 1

            if phrase in seen:
                duplicate_counter[phrase] += 1
            else:
                seen.add(phrase)


global_unique = len(seen)
duplicate_records = usable_rows - global_unique
duplicate_rate = (duplicate_records / usable_rows * 100) if usable_rows else 0

print("\n===== CLEAN GLOBAL SUMMARY =====")
print("Files used:", len(valid_files))
print("Total rows:", total_rows)
print("Usable rows:", usable_rows)
print("Global unique:", global_unique)
print("Duplicate records:", duplicate_records)
print("Duplicate rate:", round(duplicate_rate, 2), "%")
print("Bad JSON:", bad_json)
print("Empty phrase:", empty_phrase)

print("\nTop 20 duplicated phrases:")
for phrase, extra_count in duplicate_counter.most_common(20):
    print(f"{extra_count + 1}x | {phrase}")


OUT = ROOT / "phase_global_clean_analysis"
OUT.mkdir(exist_ok=True)

family_rows = []
for family, total in family_counter.items():
    unique = len(family_unique_sets[family])
    dup = total - unique
    family_rows.append({
        "family": family,
        "total_count": total,
        "unique_count": unique,
        "duplicate_count": dup,
        "duplicate_rate_percent": round((dup / total * 100) if total else 0, 2)
    })

pd.DataFrame(family_rows).sort_values(
    ["duplicate_rate_percent", "total_count"],
    ascending=[False, False]
).to_csv(OUT / "family_distribution_clean.csv", index=False)

pd.DataFrame(
    sorted(word_count_counter.items()),
    columns=["word_count", "count"]
).to_csv(OUT / "word_count_distribution_clean.csv", index=False)

pd.DataFrame(
    sorted(char_count_counter.items()),
    columns=["char_count", "count"]
).to_csv(OUT / "char_count_distribution_clean.csv", index=False)

pd.DataFrame(
    file_counter.items(),
    columns=["file", "usable_rows"]
).sort_values("file").to_csv(OUT / "files_used_clean.csv", index=False)

dupes_df = pd.DataFrame(
    duplicate_counter.most_common(100),
    columns=["passphrase", "extra_duplicate_count"]
)
dupes_df["total_occurrences"] = dupes_df["extra_duplicate_count"] + 1
dupes_df.to_csv(OUT / "top_100_duplicates_clean.csv", index=False)

summary = {
    "files_used": len(valid_files),
    "total_rows": total_rows,
    "usable_rows": usable_rows,
    "global_unique": global_unique,
    "duplicate_records": duplicate_records,
    "duplicate_rate_percent": round(duplicate_rate, 2),
    "bad_json": bad_json,
    "empty_phrase": empty_phrase
}

with open(OUT / "summary_clean.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved clean analysis outputs to:")
print(OUT)
