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

- Create a `pyproject.toml` with:

  - `[project]` metadata for `ai-media-indexer`.
  - `pydantic` as a runtime dependency.
  - `ruff` (and optionally `black`) as dev tools.
  - Ruff configured for Google-style docstrings (pydocstyle / `google` convention) and 80-character line length.

- I am not going to use ffmpeg-python. Instead, make use python's native `subprocess` to invoke it via CLI args.

  - `subprocess` over the module, because it's transparent and not abstracted.
  - Easy to catch errors, and we anyway use the direct binary.

- Only dependency we need here is `pydantic` for data validation.

- Install `ffmpeg` and `ffprobe` to your OS and test it with `ffprobe -version`.

### Task 1.2

- Goal is to create a wrapper around ffprobe so that we can invoke it via CLI programmatically.
- Create a class called `MediaProber` in `core/processing/prober.py`.
- Create a method inside it called `probe(file_path: str) -> dict`.
- The method should execute a command that

  - Outputs a JSON.
  - Make sure it's quiet. (Don't log anything to console, only capture the output for now).
  - It should return the **Streams** (Video / Audio details) with **Format** (Container details).
  - Handle errors : If file is not video or corrupted, `ffprobe` will exit with a non-zero code.
  - Our code should catch this and raise a custom exception like `MediaFileError` and not crash.

- Add Google-style docstrings to `MediaProber`, `MediaFileError`, and the `probe` method (summary line, Args, Returns, Raises).
- Topics to study :

  1. `subprocess` in python
  2. `subprocess.run` vs `subprocess.Popen`
  3. The arguments `capture_output=True` and `text=True`.
  4. FFmpeg flags: `-v quiet, -print_format json, -show_format, -show_streams`.

### Task 1.3

- Create a Class named `LibraryScanner` and write a method `scan(self, root_path: str) -> Generator[Path, None, None]` inside that class.
- The file should be `core/ingestion/scanner.py`.
- Goal: Recursively find video/audio files, ignore junk, and yield paths. (Don't return them all at once).
- This is to prevent RAM from overloading in case of terabytes of media.
- Use `pathlib.Path`, not string manipulation for paths.
- Handle `PermissionError`: If you hit a system folder you can't read.
- Add Google style formatting and docstrings for all functions/classes in this file.
- Configure `ruff` (and optionally `black`) to auto-format these files using:

  - 80-character line length.
  - Google-style docstrings.
  - Import sorting and basic linting (unused imports, unreachable code, etc.).

- Removed dead code from the project.

## Sprint 2

### Task 2.1

- Add a transcriber file to `core/processing/transcriber.py`.
