import os
import pandas as pd
import numpy as np
from pathlib import Path
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

DATA_DIR = Path("data")
SCENES_DIR = Path("data/scenes")
scenes_df = pd.read_csv(DATA_DIR / "scene_metadata.csv")

print(f"Labeling {len(scenes_df)} scenes...")


def label_scene_with_gemini(image_path):
    """Extract mood/objects/style per GACS spec"""
    img = genai.upload_file(str(image_path))

    prompt = """
    You are labeling a single video frame for an affective video dataset.
    
    Return ONLY valid JSON in this exact structure (the values below are EXAMPLES ONLY):
    
    {
      "mood": ["dark", "tense", "mysterious"],
      "objects": ["sword", "castle", "knight"],
      "style": ["cinematic", "medieval", "dramatic"]
    }
    
    Rules:
    - "mood": 3–5 adjectives describing the mood/feeling of THIS image.
    - "objects": 3–5 nouns naming the main objects visible in THIS image.
    - "style": 3 adjectives describing the visual style (lighting, color, camera, era, etc.).
    - Use words that fit THIS image, not the examples.
    - All values must be lowercase English words.
    - Do NOT include any extra keys.
    - Do NOT include any text before or after the JSON.
    """

    response = model.generate_content([prompt, img])
    raw = response.text.strip()
    print("RAW RESPONSE:\n", raw)  # keep for now; remove later if noisy

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        return {
            "mood": data.get("mood", []),
            "objects": data.get("objects", []),
            "style": data.get("style", []),
        }
    except json.JSONDecodeError as e:
        print("JSON ERROR:", e)
        return {"mood": [], "objects": [], "style": []}

labels_list = []
for idx, row in scenes_df.iterrows():
    print(f"Labeling {row['video_name']} scene {row['scene_id']}...")

    img_path = Path(row['rep_frame_path'])
    if img_path.exists():
        labels = label_scene_with_gemini(img_path)
        labels_list.append({
            **row.to_dict(),
            'mood_json': labels['mood'],
            'objects_json': labels['objects'],
            'style_json': labels['style'],
            'mood_str': ' '.join(labels['mood']),
            'objects_str': ' '.join(labels['objects']),
            'style_str': ' '.join(labels['style'])
        })
    else:
        print(f"Missing image: {img_path}")

print("Computing embeddings...")
labeled_df = pd.DataFrame(labels_list)

sentences = []
for idx, row in labeled_df.iterrows():
    sentence = f"{row['mood_str']} {row['objects_str']} {row['style_str']}"
    sentences.append(sentence)

embeddings = embedder.encode(sentences)

labeled_df['embedding'] = list(embeddings)
labeled_df['combined_text'] = sentences

labeled_df.to_csv(DATA_DIR / "gacs_labeled_scenes.csv", index=False)
np.save(DATA_DIR / "gacs_embeddings.npy", embeddings)

print(f"\nGACS Stage 3 COMPLETE!")
print(f"{len(labeled_df)} labeled scenes with embeddings")
print("data/gacs_labeled_scenes.csv")
print("data/gacs_embeddings.npy")
print("\nReady for Stage 4: Coordinate Modeling!")
