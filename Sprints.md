# Sprints for the Project

## Sprint 1

### Task 1.1

- Initialize Project. Maybe use something like `uv` or `poetry` to create a new project.
- `pip install uv` to add uv.
- Initialize this tree.

```
/ai-media-indexer
├── /core
│   ├── /ingestion     # Scanning files
│   ├── /processing    # FFmpeg & AI logic
│   └── /storage       # Database logic
├── /tests
└── main.py
```

- Goal is to create a wrapper around ffprobe.
- I am not going to use ffmpeg-python. Instead, make use python's native `subprocess` to invoke it via CLI args.
    - `subprocess` over the module, because it's transparent and not abstracted.
    - Easy to catch errors, and we anyway use the direct binary.

- Only dependency we need here is `pydantic` for data validation.
- Install `ffmpeg` and `ffprobe` to your OS and test it with `ffprobe -version`.

