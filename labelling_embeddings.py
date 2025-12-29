import os
import pandas as pd
import numpy as np
from pathlib import Path
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import re

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

DATA_DIR = Path("data")
SCENES_DIR = Path("data/scenes")
scene_meta_path = DATA_DIR / "scene_metadata.csv"

if not scene_meta_path.exists():
    print("scene_metadata.csv not found. Run frame_extraction.py first.")
    raise SystemExit(1)

scenes_df = pd.read_csv(scene_meta_path)

labeled_path = DATA_DIR / "gacs_labeled_scenes.csv"
if labeled_path.exists():
    labeled_df = pd.read_csv(labeled_path)
    labeled_set = set(zip(labeled_df["video_name"], labeled_df["scene_id"]))
    new_scenes = scenes_df[~scenes_df.apply(lambda row: (row["video_name"], row["scene_id"]) in labeled_set, axis=1)]
    print(f"Found {len(scenes_df)} total scenes, {len(new_scenes)} new to label")
else:
    new_scenes = scenes_df
    print(f"Labeling all {len(new_scenes)} new scenes")


def extract_json_from_response(raw_response: str) -> dict:
    """Robust JSON extraction from Gemini response."""
    if not raw_response:
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    if isinstance(raw_response, list):
        raw_response = raw_response[0] if raw_response else ""

    raw_response = str(raw_response).strip()

    # Method 1: Extract from `````` blocks
    json_match = re.search(r'``````', raw_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Method 2: Extract largest JSON object
        json_match = re.search(r'\{[^{}]*"[^"]*"\s*:\s*\[[^\]]*\][^{}]*\}', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Method 3: Extract first complete JSON object
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_response

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"JSON parse failed: {json_str[:200]}...")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}


def label_scene_with_gemini(image_path: Path):
    """Extract mood/objects/style per GACS spec."""
    if not image_path.exists():
        print(f"Missing: {image_path}")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    try:
        img = genai.upload_file(str(image_path))
    except Exception as e:
        print(f"Upload failed {image_path.name}: {e}")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    prompt = """
You are labeling a single video frame for an affective video dataset.

Return ONLY valid JSON in this exact structure (examples only):

{
  "mood": ["dark", "tense", "mysterious"],
  "objects": ["sword", "castle", "knight"],
  "style": ["cinematic", "medieval", "dramatic"]
}

Rules:
- mood: 3–5 adjectives describing THIS image's mood/feeling
- objects: 3–5 nouns naming main visible objects  
- style: 3 adjectives describing visual style (lighting, color, etc.)
- lowercase English words only
- NO extra text/keys/markdown
"""

    try:
        response = model.generate_content([prompt, img])
        parsed_data = extract_json_from_response(response.text)
    except Exception as e:
        print(f"Gemini API error {image_path.name}: {e}")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    mood = parsed_data.get("mood", [])
    objects = parsed_data.get("objects", [])
    style = parsed_data.get("style", [])

    return {
        "mood": mood,
        "objects": objects,
        "style": style,
        "label_ok": bool(mood and objects and style),
    }


labels_list = []
for idx, row in new_scenes.iterrows():
    print(f"[{idx + 1}/{len(new_scenes)}] {row['video_name']} scene {row['scene_id']}")

    img_path = Path(row["rep_frame_path"])
    labels = label_scene_with_gemini(img_path)

    labels_list.append({
        **row.to_dict(),
        "mood_json": labels["mood"],
        "objects_json": labels["objects"],
        "style_json": labels["style"],
        "mood_str": " ".join(labels["mood"]),
        "objects_str": " ".join(labels["objects"]),
        "style_str": " ".join(labels["style"]),
        "label_ok": labels["label_ok"],
    })

print(f"\nComputing embeddings for {len(labels_list)} new scenes...")


new_labeled_df = pd.DataFrame(labels_list)
sentences = [f"{row['mood_str']} {row['objects_str']} {row['style_str']}".strip() or "neutral scene"
             for _, row in new_labeled_df.iterrows()]
embeddings = embedder.encode(sentences)

new_labeled_df["embedding"] = list(embeddings)
new_labeled_df["combined_text"] = sentences

# === INCREMENTAL MERGE ===
labeled_path = DATA_DIR / "gacs_labeled_scenes.csv"
embeddings_path = DATA_DIR / "gacs_embeddings.npy"

if labeled_path.exists():
    print("Merging with existing labels...")
    old_df = pd.read_csv(labeled_path)

    if "embedding" in old_df.columns:
        old_df = old_df.drop(columns=["embedding"])

    # Merge + dedupe
    merged_df = pd.concat([old_df, new_labeled_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(
        subset=["video_name", "scene_id", "rep_frame_path"],
        keep="last"
    )
    print(f"Merged: {len(old_df)} old + {len(new_labeled_df)} new = {len(merged_df)} total")
else:
    merged_df = new_labeled_df
    print(f"New dataset: {len(merged_df)} scenes")

# Recompute embeddings for entire merged dataset
print("Final embeddings...")
all_sentences = [f"{row['mood_str']} {row['objects_str']} {row['style_str']}".strip() or "neutral scene"
                 for _, row in merged_df.iterrows()]
all_embeddings = embedder.encode(all_sentences)

merged_df["embedding"] = list(all_embeddings)
merged_df["combined_text"] = all_sentences

merged_df.to_csv(labeled_path, index=False)
np.save(embeddings_path, all_embeddings)

print(f"\nGACS Stage 3 COMPLETE!")
print(f"{len(merged_df)} total labeled scenes")
print(f"{labeled_path}")
print(f"{embeddings_path}")
print("Ready for Stage 4!")
