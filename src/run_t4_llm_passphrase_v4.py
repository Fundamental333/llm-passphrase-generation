import os
import re
import gc
import gzip
import json
import math
import html
import uuid
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# GLOBAL CONFIG
# ============================================================

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "/content/drive/MyDrive/LLM_Passphrase_MaxCoverage_2M/t4_llm_chunks_50k_v4"
))

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

START_CHUNK = int(os.environ.get("START_CHUNK", os.environ.get("CHUNK_ID", "0")))
END_CHUNK = int(os.environ.get("END_CHUNK", str(START_CHUNK)))

TARGET_PER_FAMILY = int(os.environ.get("TARGET_PER_FAMILY", "1250"))
N_PER_PROMPT = int(os.environ.get("N_PER_PROMPT", "12"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "180"))

GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", "20260622"))
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0") == "1"
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "2000"))

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Do not run this on CPU.")

torch.backends.cuda.matmul.allow_tf32 = True

CHUNK_DIR = OUTPUT_DIR / "chunks"
STATE_DIR = OUTPUT_DIR / "state"
MANIFEST_DIR = OUTPUT_DIR / "manifests"

for d in [OUTPUT_DIR, CHUNK_DIR, STATE_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROMPT FAMILY DESIGN
# 40 families × 1250 = 50,000 raw candidates per chunk
# ============================================================

PROMPT_FAMILIES = [
    {"family":"direct_4_plain", "mode":"spaces", "language":"English", "target_words":4, "theme":"ordinary concrete words", "shape":"exactly four separate lowercase words with spaces"},
    {"family":"direct_5_plain", "mode":"spaces", "language":"English", "target_words":5, "theme":"ordinary concrete words", "shape":"exactly five separate lowercase words with spaces"},
    {"family":"direct_6_plain", "mode":"spaces", "language":"English", "target_words":6, "theme":"ordinary concrete words", "shape":"exactly six separate lowercase words with spaces"},

    {"family":"story_memory_4", "mode":"spaces", "language":"English", "target_words":4, "theme":"tiny imaginary memory scene", "shape":"four separate words with spaces"},
    {"family":"story_memory_6", "mode":"spaces", "language":"English", "target_words":6, "theme":"tiny imaginary memory scene", "shape":"six separate words with spaces"},
    {"family":"object_scene", "mode":"spaces", "language":"English", "target_words":5, "theme":"objects in a room", "shape":"five separate words with spaces"},
    {"family":"action_scene", "mode":"spaces", "language":"English", "target_words":5, "theme":"ordinary action scene", "shape":"five separate words with spaces"},
    {"family":"adjective_noun_chain", "mode":"spaces", "language":"English", "target_words":4, "theme":"adjective noun adjective noun", "shape":"four separate words with spaces"},
    {"family":"noun_verb_noun", "mode":"spaces", "language":"English", "target_words":4, "theme":"noun verb adjective noun", "shape":"four separate words with spaces"},
    {"family":"sentence_fragment", "mode":"spaces", "language":"English", "target_words":6, "theme":"short sentence fragment", "shape":"six separate words with spaces"},

    {"family":"absurd_visual", "mode":"spaces", "language":"English", "target_words":5, "theme":"absurd but safe visual image", "shape":"five separate words with spaces"},
    {"family":"everyday_tech_objects", "mode":"spaces", "language":"English", "target_words":5, "theme":"ordinary technology objects, not cybersecurity", "shape":"five separate words with spaces"},
    {"family":"travel", "mode":"spaces", "language":"English", "target_words":5, "theme":"travel scenes", "shape":"five separate words with spaces"},
    {"family":"food", "mode":"spaces", "language":"English", "target_words":5, "theme":"food and cooking", "shape":"five separate words with spaces"},
    {"family":"sports", "mode":"spaces", "language":"English", "target_words":5, "theme":"ordinary sports moments", "shape":"five separate words with spaces"},
    {"family":"nature", "mode":"spaces", "language":"English", "target_words":5, "theme":"nature scenes", "shape":"five separate words with spaces"},
    {"family":"office", "mode":"spaces", "language":"English", "target_words":5, "theme":"office and study objects", "shape":"five separate words with spaces"},
    {"family":"household", "mode":"spaces", "language":"English", "target_words":5, "theme":"home and household items", "shape":"five separate words with spaces"},
    {"family":"music", "mode":"spaces", "language":"English", "target_words":5, "theme":"music practice and instruments", "shape":"five separate words with spaces"},
    {"family":"history_style", "mode":"spaces", "language":"English", "target_words":5, "theme":"old objects and historical settings", "shape":"five separate words with spaces"},
    {"family":"science", "mode":"spaces", "language":"English", "target_words":5, "theme":"science classroom objects", "shape":"five separate words with spaces"},
    {"family":"fantasy_plain", "mode":"spaces", "language":"English", "target_words":5, "theme":"generic fantasy imagery, no franchise names", "shape":"five separate words with spaces"},

    {"family":"code_mixed_en_hi", "mode":"spaces", "language":"English plus romanized Hindi", "target_words":5, "theme":"daily life", "shape":"five romanized ASCII words with spaces"},
    {"family":"code_mixed_en_es", "mode":"spaces", "language":"English plus Spanish", "target_words":5, "theme":"daily life", "shape":"five ASCII words with spaces, no accents"},

    {"family":"alliterative", "mode":"spaces", "language":"English", "target_words":4, "theme":"same starting sound", "shape":"four separate words with spaces"},
    {"family":"rhyming_light", "mode":"spaces", "language":"English", "target_words":4, "theme":"light sound pattern", "shape":"four separate words with spaces"},
    {"family":"compound_split", "mode":"spaces", "language":"English", "target_words":4, "theme":"compound-like word pairs", "shape":"four separate words with spaces"},

    {"family":"number_end", "mode":"number_end", "language":"English", "target_words":4, "theme":"three words plus one small number", "shape":"three words then one number at the end"},
    {"family":"number_inside", "mode":"number_inside", "language":"English", "target_words":5, "theme":"four words plus one number inside", "shape":"four words and exactly one number token"},
    {"family":"symbol_separator", "mode":"symbol", "language":"English", "target_words":4, "theme":"plain words joined by one separator", "shape":"four words joined by one repeated separator from hyphen, underscore, or dot"},
    {"family":"hyphenated", "mode":"hyphen", "language":"English", "target_words":5, "theme":"plain words joined by hyphens", "shape":"five words joined by hyphens"},
    {"family":"spaced_words", "mode":"spaces", "language":"English", "target_words":5, "theme":"plain words", "shape":"five separate words with spaces"},
    {"family":"camelcase", "mode":"camelcase", "language":"English", "target_words":4, "theme":"plain words in CamelCase", "shape":"four words joined as CamelCase with no spaces"},
    {"family":"lowercase_spaces", "mode":"spaces", "language":"English", "target_words":5, "theme":"plain lowercase words", "shape":"five lowercase words separated by spaces"},
    {"family":"mixedcase", "mode":"spaces", "language":"English", "target_words":4, "theme":"plain mixed-case words", "shape":"four separate words with spaces"},

    {"family":"high_entropy_random_words", "mode":"spaces", "language":"English", "target_words":6, "theme":"unrelated random words", "shape":"six separate words with spaces"},
    {"family":"low_entropy_humanlike", "mode":"spaces", "language":"English", "target_words":4, "theme":"human-chosen memorable phrase", "shape":"four separate words with spaces"},
    {"family":"typo_humanlike", "mode":"spaces", "language":"English", "target_words":5, "theme":"human phrase with one mild typo", "shape":"five separate words with spaces"},
    {"family":"memory_scene_phrase", "mode":"spaces", "language":"English", "target_words":6, "theme":"imaginary memory scene phrase", "shape":"six simple words with spaces, no labels or colons"},
    {"family":"few_shot_template", "mode":"spaces", "language":"English", "target_words":5, "theme":"varied human memorable phrase", "shape":"five separate words with spaces"},
]

DECODING_PROFILES = [
    {"name":"controlled", "temperature":0.70, "top_p":0.90, "top_k":40},
    {"name":"balanced", "temperature":0.85, "top_p":0.92, "top_k":50},
    {"name":"diverse", "temperature":1.00, "top_p":0.95, "top_k":70},
    {"name":"wide", "temperature":1.12, "top_p":0.97, "top_k":90},
]

SYSTEM_MSG = (
    "You generate synthetic passphrase strings for security research. "
    "Return only passphrase strings. One passphrase per line. "
    "No numbering, no bullets, no headers, no explanations."
)

GOOD_BAD = {
    "spaces": {
        "good": ["quiet orange ladder moon", "silver kettle waits outside"],
        "bad": ["aeroplane", "ordinarylakes", "DreamTeamRunnersRallying"]
    },
    "hyphen": {
        "good": ["quiet-orange-ladder-moon", "silver-kettle-window-river"],
        "bad": ["quiet orange ladder moon", "quiet_orange_ladder_moon"]
    },
    "symbol": {
        "good": ["quiet-orange-ladder-moon", "silver_kettle_window_river", "paper.river.green.chair"],
        "bad": ["quiet orange ladder moon", "quiet-orange_ladder.moon"]
    },
    "camelcase": {
        "good": ["QuietOrangeLadderMoon", "SilverGardenWindowRiver"],
        "bad": ["quiet orange ladder moon", "SecurePassword456", "CryptographicKey123"]
    },
    "number_end": {
        "good": ["quiet orange ladder 472", "silver garden window 91"],
        "bad": ["quiet orange 472 ladder", "password1234"]
    },
    "number_inside": {
        "good": ["quiet 472 orange ladder moon", "silver garden 91 window river"],
        "bad": ["quiet orange ladder moon", "password1234"]
    }
}

BAD_PATTERNS = [
    r"password", r"passphrase", r"credential", r"login", r"admin", r"root", r"secret",
    r"example", r"qwerty", r"123456", r"https?://", r"www\.", r"@\w+\.\w+",
    r"here are", r"as an ai", r"generate", r"candidate",
    r"cryptographic", r"encryption", r"hash", r"secure"
]

# ============================================================
# HELPERS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def split_camel_token(token):
    pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token)
    return pieces if pieces else [token]

def split_words(s):
    base = re.split(r"[\s\-_\.]+", s)
    out = []
    for p in base:
        p = p.strip("!?,:;()[]{}<>@#$%^&*+=/\\|'\"")
        if not p:
            continue
        for q in split_camel_token(p):
            if q and any(ch.isalnum() for ch in q):
                out.append(q)
    return out

def count_number_tokens(s):
    return len(re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", s))

def entropy_proxy_bits(s):
    words = split_words(s)
    unique_words = len(set(w.lower() for w in words))

    char_classes = 0
    char_classes += any(c.islower() for c in s)
    char_classes += any(c.isupper() for c in s)
    char_classes += any(c.isdigit() for c in s)
    char_classes += any(not c.isalnum() and not c.isspace() for c in s)

    return round((unique_words * math.log2(2048)) + (char_classes * 2.0), 2)

def clean_candidate(s):
    s = html.unescape(s or "")
    s = s.strip().strip('"\'`“”‘’')
    s = re.sub(r"^\s*(?:[-*•]+|\d{1,4}[\).:\-]+)\s*", "", s)
    s = re.sub(r"^(passphrase|candidate|answer|output|line)\s*[:\-]\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip().strip('"\'`“”‘’')

def general_validate(s):
    if not s:
        return False, "empty"

    if len(s) < 10:
        return False, "too_short"

    if len(s) > 100:
        return False, "too_long"

    low = s.lower()

    for pat in BAD_PATTERNS:
        if re.search(pat, low):
            return False, f"bad_pattern:{pat}"

    if not re.fullmatch(r"[A-Za-z0-9 \-_.!?#$%&*+]+", s):
        return False, "bad_characters"

    words = split_words(s)

    if len(words) < 3:
        return False, "too_few_words"

    if len(words) > 14:
        return False, "too_many_words"

    if len(set(w.lower() for w in words)) < max(2, len(words) - 3):
        return False, "too_repetitive_words"

    if re.search(r"(.)\1{4,}", s):
        return False, "repeated_characters"

    return True, "valid"

def format_match(s, fam):
    mode = fam["mode"]
    words = split_words(s)

    if mode == "spaces":
        return (" " in s) and ("_" not in s) and ("." not in s) and ("-" not in s) and len(words) >= 3

    if mode == "hyphen":
        return ("-" in s) and (" " not in s) and len(words) >= 3

    if mode == "symbol":
        seps = [sep for sep in ["-", "_", "."] if sep in s]
        return len(seps) == 1 and (" " not in s) and len(words) >= 3

    if mode == "camelcase":
        return (" " not in s) and ("-" not in s) and ("_" not in s) and ("." not in s) and len(words) >= 3 and any(c.isupper() for c in s)

    if mode == "number_end":
        toks = s.split()
        return len(toks) >= 4 and toks[-1].isdigit() and count_number_tokens(s) == 1

    if mode == "number_inside":
        return count_number_tokens(s) == 1 and len(words) >= 4

    return len(words) >= 3

def extract_candidates(text):
    text = text.replace("```", "\n")
    text = text.replace("\r", "\n")

    # Split one-line numbered dumps like "1. abc 2. def"
    text = re.sub(r"\s+(?=\d{1,3}[\).:\-]\s+)", "\n", text)

    # Split bullets
    text = re.sub(r"\s+(?=[-*•]\s+)", "\n", text)

    raw_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"^(here are|sure|outputs?|passphrases?|candidates?)\b", line, re.I):
            continue

        if " | " in line:
            raw_lines.extend([x.strip() for x in line.split(" | ") if x.strip()])
            continue

        if ";" in line and len(line.split(";")) <= 25:
            raw_lines.extend([x.strip() for x in line.split(";") if x.strip()])
            continue

        comma_parts = [x.strip() for x in line.split(",") if x.strip()]
        if 3 <= len(comma_parts) <= 25 and all(len(x.split()) <= 8 for x in comma_parts):
            raw_lines.extend(comma_parts)
            continue

        raw_lines.append(line)

    return raw_lines

def build_prompt(fam, profile, seed):
    mode = fam["mode"]
    examples = GOOD_BAD.get(mode, GOOD_BAD["spaces"])

    return f"""
Create exactly {N_PER_PROMPT} synthetic passphrases.

Output format:
one passphrase per line
no numbering
no bullets
no commas between entries
no explanation

Required shape:
{fam["shape"]}

Theme:
{fam["theme"]}

Language:
{fam["language"]}

Seed:
{seed}

Style profile:
{profile["name"]}

Good examples of the required shape:
{examples["good"][0]}
{examples["good"][1]}

Bad examples to avoid:
{examples["bad"][0]}
{examples["bad"][1]}
{examples["bad"][2] if len(examples["bad"]) > 2 else ""}

Strict rules:
Do not output a single word.
Do not join separate words together unless the required shape is CamelCase.
Do not create usernames, product names, team names, slogans, labels, titles, codes, keys, hashes, or security terms.
Do not use colons, apostrophes, quotation marks, URLs, emails, real names, or personal data.
Avoid these words: password, passphrase, login, admin, root, secret, credential, qwerty, secure, encryption, cryptographic, hash.
""".strip()

def make_chat(tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return SYSTEM_MSG + "\n\n" + prompt + "\n"

def write_jsonl(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def save_summary(path, summary):
    summary["elapsed_seconds"] = round(time.time() - summary["_start_time"], 2)
    tmp = dict(summary)
    tmp.pop("_start_time", None)
    path.write_text(json.dumps(tmp, indent=2), encoding="utf-8")

# ============================================================
# SAVE MANIFESTS
# ============================================================

(MANIFEST_DIR / "prompt_families.json").write_text(
    json.dumps(PROMPT_FAMILIES, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

(MANIFEST_DIR / "decoding_profiles.json").write_text(
    json.dumps(DECODING_PROFILES, indent=2),
    encoding="utf-8"
)

# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Loading:", MODEL_ID)
print("GPU:", torch.cuda.get_device_name(0))
print("Chunks:", START_CHUNK, "to", END_CHUNK)

random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

model.eval()

# ============================================================
# CHUNK GENERATION FUNCTION
# ============================================================

def generate_chunk(chunk_id):
    run_id = f"chunk_{chunk_id:03d}"

    raw_final = CHUNK_DIR / f"{run_id}_raw_attempts.jsonl.gz"
    valid_final = CHUNK_DIR / f"{run_id}_valid_subset.jsonl.gz"
    summary_final = CHUNK_DIR / f"{run_id}_summary.json"

    raw_tmp = CHUNK_DIR / f"{run_id}_raw_attempts.running.jsonl.gz"
    valid_tmp = CHUNK_DIR / f"{run_id}_valid_subset.running.jsonl.gz"
    summary_live = STATE_DIR / f"{run_id}_live_summary.json"

    if summary_final.exists() and not FORCE_RERUN:
        print(f"{run_id} already completed. Skipping.")
        return

    # Remove broken partial files if rerunning
    if FORCE_RERUN:
        for p in [raw_tmp, valid_tmp, summary_live]:
            if p.exists():
                p.unlink()

    random.seed(GLOBAL_SEED + chunk_id)
    torch.manual_seed(GLOBAL_SEED + chunk_id)

    summary = {
        "run_id": run_id,
        "chunk_id": chunk_id,
        "model_id": MODEL_ID,
        "device": "gpu",
        "gpu_name": torch.cuda.get_device_name(0),
        "target_per_family": TARGET_PER_FAMILY,
        "target_raw_attempts": TARGET_PER_FAMILY * len(PROMPT_FAMILIES),
        "n_prompt_families": len(PROMPT_FAMILIES),
        "n_per_prompt": N_PER_PROMPT,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "started_at_utc": now_iso(),
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "raw_attempts": 0,
        "valid_unique_within_chunk": 0,
        "valid_format_match_within_chunk": 0,
        "valid_duplicates_within_chunk": 0,
        "invalid": 0,
        "by_family": {},
        "_start_time": time.time(),
    }

    seen_in_chunk = set()
    gen_calls = 0

    print(f"\n===== Starting {run_id} =====")

    with gzip.open(raw_tmp, "wt", encoding="utf-8") as raw_f, gzip.open(valid_tmp, "wt", encoding="utf-8") as valid_f:
        for fam_idx, fam in enumerate(PROMPT_FAMILIES):
            fam_name = fam["family"]

            fam_summary = {
                "raw_attempts": 0,
                "valid_unique_within_chunk": 0,
                "valid_format_match_within_chunk": 0,
                "valid_duplicates_within_chunk": 0,
                "invalid": 0,
            }

            summary["by_family"][fam_name] = fam_summary

            pbar = tqdm(total=TARGET_PER_FAMILY, desc=f"{run_id}:{fam_name}")
            prompt_counter = 0

            while fam_summary["raw_attempts"] < TARGET_PER_FAMILY:
                batch_prompts = []
                batch_meta = []

                profile = DECODING_PROFILES[(fam_idx + chunk_id + prompt_counter) % len(DECODING_PROFILES)]

                for b in range(BATCH_SIZE):
                    seed = (
                        GLOBAL_SEED
                        + chunk_id * 1_000_000
                        + fam_idx * 10_000
                        + prompt_counter * BATCH_SIZE
                        + b
                    )

                    prompt = build_prompt(fam, profile, seed)
                    full_prompt = make_chat(tokenizer, prompt)

                    batch_prompts.append(full_prompt)
                    batch_meta.append({
                        "seed": seed,
                        "prompt_hash": sha256_text(prompt),
                        "profile": profile,
                    })

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(model.device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=profile["temperature"],
                        top_p=profile["top_p"],
                        top_k=profile["top_k"],
                        max_new_tokens=MAX_NEW_TOKENS,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_only = outputs[:, inputs["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

                prompt_counter += 1
                gen_calls += 1

                for out_i, text in enumerate(texts):
                    for raw_line in extract_candidates(text):
                        if fam_summary["raw_attempts"] >= TARGET_PER_FAMILY:
                            break

                        cleaned = clean_candidate(raw_line)
                        is_valid, reason = general_validate(cleaned)
                        fmt_match = format_match(cleaned, fam) if is_valid else False
                        words = split_words(cleaned)

                        rec = {
                            "sample_id": str(uuid.uuid4()),
                            "generator_type": "llm",
                            "model_id": MODEL_ID,
                            "device": "gpu",
                            "gpu_name": torch.cuda.get_device_name(0),
                            "chunk_id": chunk_id,
                            "prompt_family": fam_name,
                            "prompt_mode": fam["mode"],
                            "prompt_hash": batch_meta[out_i]["prompt_hash"],
                            "decoding_profile": batch_meta[out_i]["profile"]["name"],
                            "temperature": batch_meta[out_i]["profile"]["temperature"],
                            "top_p": batch_meta[out_i]["profile"]["top_p"],
                            "top_k": batch_meta[out_i]["profile"]["top_k"],
                            "seed": batch_meta[out_i]["seed"],
                            "language": fam["language"],
                            "target_word_count": fam["target_words"],
                            "theme": fam["theme"],
                            "raw_output": raw_line,
                            "cleaned_passphrase": cleaned,
                            "word_count": len(words),
                            "char_count": len(cleaned),
                            "format_match": fmt_match,
                            "entropy_theoretical": None,
                            "entropy_proxy_bits": entropy_proxy_bits(cleaned) if cleaned else None,
                            "filter_status": "valid" if is_valid else "invalid",
                            "failure_reason": reason if not is_valid else None,
                            "timestamp_utc": now_iso(),
                        }

                        fam_summary["raw_attempts"] += 1
                        summary["raw_attempts"] += 1
                        pbar.update(1)

                        if is_valid:
                            h = sha256_text(cleaned.lower())

                            if h in seen_in_chunk:
                                rec["filter_status"] = "duplicate_valid_within_chunk"
                                fam_summary["valid_duplicates_within_chunk"] += 1
                                summary["valid_duplicates_within_chunk"] += 1
                            else:
                                seen_in_chunk.add(h)
                                write_jsonl(valid_f, rec)
                                fam_summary["valid_unique_within_chunk"] += 1
                                summary["valid_unique_within_chunk"] += 1

                                if fmt_match:
                                    fam_summary["valid_format_match_within_chunk"] += 1
                                    summary["valid_format_match_within_chunk"] += 1
                        else:
                            fam_summary["invalid"] += 1
                            summary["invalid"] += 1

                        write_jsonl(raw_f, rec)

                        if summary["raw_attempts"] % SAVE_EVERY == 0:
                            save_summary(summary_live, summary)

                del inputs, outputs, gen_only

                if gen_calls % 20 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            pbar.close()
            save_summary(summary_live, summary)

    summary["finished_at_utc"] = now_iso()
    save_summary(summary_live, summary)

    raw_tmp.replace(raw_final)
    valid_tmp.replace(valid_final)
    summary_live.replace(summary_final)

    final_summary = dict(summary)
    final_summary.pop("_start_time", None)

    print(json.dumps(final_summary, indent=2))
    print(f"===== Finished {run_id} =====")

# ============================================================
# RUN REQUESTED CHUNKS
# ============================================================

for cid in range(START_CHUNK, END_CHUNK + 1):
    generate_chunk(cid)

print("All requested chunks complete.")
