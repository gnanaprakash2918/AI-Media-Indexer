# AI-Media-Indexer LLM Coding Standards

> **Purpose**: This document contains mandatory coding patterns, dependency management rules, and anti-patterns that all LLM agents must follow when working on this codebase. Following these rules prevents regressions and maintains code quality.

---

## 1. Type Annotations & Type Ignores

### When `# type: ignore` is REQUIRED

Use `# type: ignore` ONLY for these specific cases:

```python
# ✅ Optional imports (packages that may not be installed)
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None

# ✅ Incomplete library type stubs (Qdrant, OpenCV, etc.)
points = ids  # type: ignore  # Qdrant expects ExtendedPointId
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

# ✅ Dynamic ML model APIs
model.to("cuda")  # type: ignore  # Dynamic method
model.freeze()  # type: ignore  # NeMo model method

# ✅ Platform-specific APIs
temps = psutil.sensors_temperatures()  # type: ignore  # Linux-only
```

### When `# type: ignore` is FORBIDDEN

```python
# ❌ NEVER use type:ignore to suppress fixable type errors
result: Any = some_function()  # type: ignore  # BAD - fix the type instead

# ❌ NEVER ignore return type mismatches
def get_data() -> str:
    return None  # type: ignore  # BAD - fix the return type
```

### Proper Type Annotation Patterns

```python
# ✅ Use union types for nullable returns
def get_user() -> User | None:
    return None

# ✅ Use TypeVar for generic wrappers
T = TypeVar('T')
def wrapper(func: Callable[..., T]) -> Callable[..., T]:
    ...

# ✅ Use Annotated for FastAPI dependencies (not default args)
def endpoint(db: Annotated[VectorDB, Depends(get_db)]):
    ...
```

---

## 2. Dependency Management

### Required Dependencies That MUST Be Installed

| Package | Version | Purpose |
|---------|---------|---------|
| `tf-keras` | >=2.16.0 | Required for transformers with Keras 3 |
| `pyloudnorm` | >=0.1.1 | Audio loudness normalization |
| `fer` | >=22.5.1 | Facial emotion recognition |
| `deepface` | >=0.0.93 | Face analysis |
| `ultralytics` | >=8.3.0 | YOLO object detection |
| `paddleocr` | >=2.9.1 | OCR engine |
| `FlagEmbedding` | >=1.3.3 | BGE embeddings |
| `rank_bm25` | >=0.2.2 | BM25 keyword search |

### Optional Dependencies (Graceful Fallback Required)

These packages may not be installed. Code MUST handle `ImportError`:

```python
# ✅ Correct pattern for optional dependencies
try:
    from inaSpeechSegmenter import Segmenter
    HAS_INA = True
except ImportError:
    HAS_INA = False
    Segmenter = None  # type: ignore

def classify(audio_path: Path) -> list[Region]:
    if HAS_INA:
        return _classify_with_ina(audio_path)
    return _fallback_classify(audio_path)
```

| Optional Package | Fallback Behavior |
|------------------|-------------------|
| `inaSpeechSegmenter` | Use energy-based fallback |
| `brave-search` | Return empty results, log warning |
| `nemo-toolkit` | Use Whisper for transcription |
| `scenedetect` | Skip scene detection |
| `sam2` | Use ultralytics SAM |

### Version Constraints

```toml
# ✅ Always specify upper bounds for ML packages
"ml_dtypes>=0.4.0,<0.6.0"  # Prevents breaking changes
"numpy>=1.26.0,<2.0.0"      # NumPy 2.0 breaks many packages
"transformers==4.46.3"       # Pin transformers exactly
```

---

## 3. Ruff & Linting Rules

### Ignored Rules (Configured in pyproject.toml)

| Rule | Reason |
|------|--------|
| D203 | Conflicts with D211 (blank line before class docstring) |
| D213 | Conflicts with D212 (multi-line docstring summary position) |
| E501 | Line length handled by Black |
| B008 | FastAPI uses mutable default arguments by design |

### Common Fixes for Ruff Errors

```python
# B904: Raise from original exception
except ValueError as e:
    raise RuntimeError("message") from e  # ✅ Not just 'raise'

# N806: Variable should be lowercase
CONSTANT_VALUE = 10  # Module-level OK
def func():
    local_value = 10  # ✅ Not LOCAL_VALUE

# C401: Unnecessary generator (use set comprehension)
unique = {x.id for x in items}  # ✅ Not set(x.id for x in items)

# E722: Do not use bare except
except Exception:  # ✅ Not 'except:'
```

---

## 4. Qdrant Client Patterns

### Common Type Issues

```python
# Qdrant models need type: ignore for some parameters
from qdrant_client import models

# ✅ Filter with type ignore (incomplete stubs)
result = client.scroll(
    collection_name="frames",
    scroll_filter=models.Filter(must=conditions),  # type: ignore
)

# ✅ Point IDs need type ignore
client.delete(
    collection_name="faces",
    points=point_ids,  # type: ignore
)

# ✅ MatchAny with type ignore  
models.FieldCondition(
    key="cluster_id",
    match=models.MatchAny(any=[cluster_id]),  # type: ignore
)
```

---

## 5. FastAPI & Pydantic Patterns

### Dependency Injection

```python
# ✅ Use Annotated for path/query params with validators
from typing import Annotated
from fastapi import Depends, Query

async def search(
    query: Annotated[str, Query(min_length=1)],
    db: Annotated[VectorDB, Depends(get_db)],
    limit: int = 10,  # Simple defaults still OK
):
    ...

# ❌ Don't silence B008 when Annotated works
async def bad_endpoint(
    file: UploadFile = File(...),  # noqa: B008  # Avoid if possible
):
    ...
```

### Pydantic Models

```python
from pydantic import BaseModel, Field

# ✅ Use Field for defaults and validation
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    
# ✅ Use model_dump() not dict() (Pydantic v2)
data = request.model_dump()
```

---

## 6. Async Patterns

### Proper Error Handling

```python
# ✅ Use 'from e' to preserve stack trace
async def process():
    try:
        await risky_operation()
    except ValueError as e:
        raise ProcessingError("Failed") from e

# ✅ Use asyncio.gather with return_exceptions for parallel ops
results = await asyncio.gather(
    task1(), task2(), task3(),
    return_exceptions=True
)
for result in results:
    if isinstance(result, Exception):
        log.error(f"Task failed: {result}")
```

### Resource Cleanup

```python
# ✅ Always close async clients
class Service:
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

# ✅ Use context managers when available
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

---

## 7. ML Model Patterns

### Lazy Loading

```python
# ✅ Lazy load models to reduce startup time
class Processor:
    _model = None
    
    def _load_model(self):
        if Processor._model is None:
            Processor._model = load_heavy_model()
        return Processor._model
```

### Memory Management

```python
# ✅ Move models between CPU/GPU as needed
def offload_to_cpu(self):
    if self.model is not None:
        self.model.to("cpu")  # type: ignore
        torch.cuda.empty_cache()
```

---

## 8. Logging Patterns

```python
from core.utils.logger import log, get_logger

# ✅ Use component prefix in log messages
log("[VectorDB] Indexing 1000 frames")
log("[Transcriber] Processing audio segment")

# ✅ Include relevant context
log(f"[Pipeline] Video {video_path.name}: extracted {len(frames)} frames")

# ❌ Don't use print() for logging
print("Debug message")  # BAD - use log()
```

---

## 9. File Path Patterns

```python
from pathlib import Path

# ✅ Always use Path objects, not strings
audio_path = Path(audio_file)
output_path = thumbnails_dir / f"{frame_id}.jpg"

# ✅ Check existence before operations
if audio_path.exists():
    process(audio_path)

# ✅ Use resolve() for absolute paths
abs_path = relative_path.resolve()
```

---

## 10. Anti-Patterns to AVOID

### ❌ Never Do These

```python
# ❌ Bare except clause
try:
    risky()
except:  # BAD
    pass

# ❌ Mutable default arguments (except FastAPI Depends)
def func(items: list = []):  # BAD
    items.append(1)

# ❌ String concatenation for paths
path = folder + "/" + filename  # BAD - use Path /

# ❌ Ignoring return values
db.update(data)  # BAD if update returns success status

# ❌ Using deprecated APIs
model.dict()  # BAD - use model.model_dump()
```

---

## Quick Reference Checklist

Before committing, verify:

- [ ] All `# type: ignore` comments are for legitimate cases listed above
- [ ] No bare `except:` clauses (use `except Exception:`)
- [ ] All exceptions re-raised use `from e` pattern
- [ ] Optional dependencies wrapped in try/except with fallback
- [ ] No mutable default arguments (except FastAPI Depends)
- [ ] Using `Path` objects, not string paths
- [ ] All async resources properly closed
- [ ] Log messages include component prefix
