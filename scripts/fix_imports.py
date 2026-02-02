
import os
import re

REPLACEMENTS = [
    # AUDIO
    (r"core\.processing\.transcriber", "core.processing.audio.transcriber"),
    (r"core\.processing\.voice", "core.processing.audio.voice"),
    (r"core\.processing\.audio_events", "core.processing.audio.audio_events"),
    (r"core\.processing\.indic_transcriber", "core.processing.audio.indic_transcriber"),
    (r"core\.processing\.content_classifier", "core.processing.audio.content_classifier"),
    (r"core\.processing\.speech_emotion", "core.processing.audio.speech_emotion"),
    
    # VISION
    (r"core\.processing\.ocr", "core.processing.vision.vision.ocr"),
    (r"core\.processing\.vision\b(?!\.vision)", "core.processing.vision.vision"), # Only replace .vision if not followed by .vision (avoid double replace)
    (r"core\.processing\.visual_encoder", "core.processing.vision.vision.visual_encoder"),
    (r"core\.processing\.scene_detector", "core.processing.vision.vision.scene_detector"),
    (r"core\.processing\.extractor", "core.processing.vision.vision.extractor"),
    (r"core\.processing\.segmentation", "core.processing.vision.vision.segmentation"),
    (r"core\.processing\.identity", "core.processing.vision.vision.identity"),
    (r"core\.processing\.frame_sampling", "core.processing.vision.vision.frame_sampling"),
    (r"core\.processing\.clock_reader", "core.processing.vision.vision.clock_reader"),
    (r"core\.processing\.content_moderation", "core.processing.vision.vision.content_moderation"),

    # ANALYSIS
    (r"core\.processing\.deep_research", "core.processing.analysis.deep_research"),
    (r"core\.processing\.cinematography", "core.processing.analysis.cinematography"),
    (r"core\.processing\.audio_analysis", "core.processing.analysis.audio_analysis"),
    (r"core\.processing\.temporal_context", "core.processing.analysis.temporal_context"),
    (r"core\.processing\.text_utils", "core.processing.analysis.text_utils"),

    # METADATA
    (r"core\.processing\.metadata\b(?!\.metadata)", "core.processing.metadata.metadata"),
    (r"core\.processing\.prober", "core.processing.metadata.metadata.prober"),
]

def fix_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = content
        for pattern, replacement in REPLACEMENTS:
            # Use regex to replace ensuring boundaries or literal context
            # Simple replace might be risky for partial matches but import paths are usually distinct
            new_content = re.sub(pattern, replacement, new_content)

        if new_content != content:
            print(f"Fixing imports in: {filepath}")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    root_dir = os.getcwd()
    print(f"Scanning {root_dir}...")
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        # Skip some dirs
        if "node_modules" in dirs: dirs.remove("node_modules")
        if ".git" in dirs: dirs.remove(".git")
        if ".venv" in dirs: dirs.remove(".venv")
        if "__pycache__" in dirs: dirs.remove("__pycache__")
        
        for file in files:
            if file.endswith(".py"):
                fix_file(os.path.join(root, file))
                count += 1
    print(f"Scanned {count} python files.")

if __name__ == "__main__":
    main()
