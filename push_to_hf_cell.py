# ── Push to Hugging Face Space ───────────────────────────────────────────
# Run this cell AFTER training finishes (Cell 5 complete)
# Only needs to run once — after that the Space is live

HF_USERNAME = "Bindkushal"
HF_SPACE    = "the-child"
HF_TOKEN    = "hf_XXXXXXXXXXXXXXXXXXXX"   # ← paste your token here

import subprocess, shutil, os
from pathlib import Path

# 1. Install HF hub
subprocess.run(["pip", "install", "huggingface_hub", "--quiet"], check=True)
from huggingface_hub import HfApi, create_repo

api = HfApi()

# 2. Create Space if it doesn't exist yet
repo_id = f"{HF_USERNAME}/{HF_SPACE}"
try:
    create_repo(
        repo_id   = repo_id,
        repo_type = "space",
        space_sdk = "gradio",
        private   = False,
        token     = HF_TOKEN,
        exist_ok  = True,
    )
    print(f"✓ Space ready: huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"Space creation: {e}")

# 3. Files to upload
files_to_upload = {
    "app.py":                    "app.py",
    "requirements.txt":          "requirements.txt",
    "dynamic_net.py":            "dynamic_net.py",
    "senn_final.pt":             "senn_final.pt",
    "architecture_final.json":   "architecture_final.json",
}

# Also upload growth_journal if it exists
if os.path.exists("growth_journal.json"):
    files_to_upload["growth_journal.json"] = "growth_journal.json"

# 4. Upload each file
print("\nUploading files...")
for local_path, remote_path in files_to_upload.items():
    if not os.path.exists(local_path):
        print(f"  ⚠ Skipping {local_path} — not found")
        continue
    try:
        api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = remote_path,
            repo_id         = repo_id,
            repo_type       = "space",
            token           = HF_TOKEN,
        )
        size = os.path.getsize(local_path)
        print(f"  ✓ {local_path} ({size/1024:.1f} KB)")
    except Exception as e:
        print(f"  ✗ {local_path} failed: {e}")

print(f"\n✓ Done — Space live at:")
print(f"  https://huggingface.co/spaces/{repo_id}")
