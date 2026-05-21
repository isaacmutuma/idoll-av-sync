"""JSON database for enrolled face and voice embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DATABASE_FILENAME = "database.json"
FACE_EMBEDDING_DIM = 512
VOICE_EMBEDDING_DIM = 256


class DatabaseError(Exception):
    """Raised when the embedding database cannot be read or written."""


def get_database_path() -> Path:
    """
    Return the absolute path to ``enrollment/database.json``.

    The file lives next to this module so imports work regardless of cwd.
    """
    return Path(__file__).resolve().parent / DATABASE_FILENAME


def load_database(path: Path | None = None) -> dict[str, Any]:
    """
    Load the enrollment database from disk.

    Returns an empty dict if the file does not exist yet.
    """
    db_path = path or get_database_path()
    if not db_path.exists():
        return {}

    try:
        with db_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise DatabaseError(f"Invalid JSON in {db_path}") from exc
    except OSError as exc:
        raise DatabaseError(f"Cannot read {db_path}") from exc

    if not isinstance(data, dict):
        raise DatabaseError(f"Expected a JSON object at root of {db_path}")

    return data


def _validate_embedding(
    embedding: np.ndarray,
    expected_dim: int,
    label: str,
) -> list[float]:
    """Ensure embedding is a 1-D float vector of the expected length."""
    array = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if array.shape[0] != expected_dim:
        raise DatabaseError(
            f"{label} embedding must be {expected_dim}-d, got {array.shape[0]}"
        )
    return array.tolist()


def save_person(
    name: str,
    face_embedding: np.ndarray,
    voice_embedding: np.ndarray,
    path: Path | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """
    Persist one person's face and voice embeddings under their name.

    Args:
        name: Unique identifier for the enrolled person.
        face_embedding: 128-dimensional face vector.
        voice_embedding: 256-dimensional voice vector.
        path: Optional override for the database file location.
        overwrite: If False, raise when ``name`` already exists.
    """
    cleaned_name = name.strip()
    if not cleaned_name:
        raise DatabaseError("Person name cannot be empty")

    db_path = path or get_database_path()
    database = load_database(db_path)

    if cleaned_name in database and not overwrite:
        raise DatabaseError(f"'{cleaned_name}' is already enrolled")

    database[cleaned_name] = {
        "face_embedding": _validate_embedding(
            face_embedding, FACE_EMBEDDING_DIM, "Face"
        ),
        "voice_embedding": _validate_embedding(
            voice_embedding, VOICE_EMBEDDING_DIM, "Voice"
        ),
    }

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_suffix(".json.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(database, file, indent=2)
            file.write("\n")
        temp_path.replace(db_path)
    except OSError as exc:
        raise DatabaseError(f"Cannot write {db_path}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def list_enrolled_names(path: Path | None = None) -> list[str]:
    """Return sorted names of everyone currently in the database."""
    return sorted(load_database(path).keys())


def person_exists(name: str, path: Path | None = None) -> bool:
    """Return True if ``name`` (after stripping) is already enrolled."""
    return name.strip() in load_database(path)


if __name__ == "__main__":
    import tempfile

    print("Testing enrollment.database ...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "test_database.json"
        face = np.random.randn(FACE_EMBEDDING_DIM)
        voice = np.random.randn(VOICE_EMBEDDING_DIM)

        save_person("Test User", face, voice, path=test_path)
        assert person_exists("Test User", path=test_path)
        loaded = load_database(test_path)
        assert len(loaded["Test User"]["face_embedding"]) == FACE_EMBEDDING_DIM
        assert len(loaded["Test User"]["voice_embedding"]) == VOICE_EMBEDDING_DIM
        assert list_enrolled_names(test_path) == ["Test User"]
        print("All database tests passed.")
