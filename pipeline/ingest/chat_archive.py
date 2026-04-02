"""
Ingest ChatGPT conversation archives into the governed pipeline.

Reads individual conversation JSON files (theory-archive format)
or the bulk conversations.json from ChatGPT data export.

Each message becomes a candidate Segment for S2 classification.
Technical messages (MaL reasoning, governance proofs, operator derivations)
proceed to S3 decomposition. Non-technical messages are excluded.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterator

from pipeline.stages.s1_segment import Segment
from pipeline.types import SourceProvenance, Tier


def ingest_conversation_file(path: str | Path) -> Iterator[Segment]:
    """Ingest a single conversation JSON file (theory-archive format)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        conv = json.load(f)

    file_hash = sha256(path.read_bytes()).hexdigest()
    title = conv.get("title", path.stem)
    conv_id = conv.get("conversation_id", path.stem)
    create_time = conv.get("create_time")
    timestamp = (
        datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat()
        if create_time else "unknown"
    )

    provenance = SourceProvenance(
        source_id=f"chat:{conv_id}",
        tier=Tier.T3,
        url=f"theory-archive/{path.name}",
        commit_or_version=timestamp,
        license="proprietary",
        acquired_at=datetime.now(timezone.utc).isoformat(),
        artifact_sha256=file_hash,
    )

    # ChatGPT format: mapping dict with message nodes
    mapping = conv.get("mapping", {})
    offset = 0

    for node_id, node in mapping.items():
        msg = node.get("message")
        if not msg:
            continue

        role = msg.get("author", {}).get("role", "unknown")
        if role == "system":
            continue  # Skip system prompts

        content = msg.get("content", {})
        parts = content.get("parts", [])
        text = ""
        for part in parts:
            if isinstance(part, str):
                text += part

        text = text.strip()
        if not text:
            continue

        msg_id = msg.get("id", node_id)
        byte_start = offset
        byte_end = offset + len(text.encode("utf-8"))
        offset = byte_end

        yield Segment(
            source=provenance,
            byte_range=(byte_start, byte_end),
            text=text,
            segment_type=f"chat_{role}",
        )


def ingest_conversation_dir(dir_path: str | Path) -> Iterator[Segment]:
    """Ingest all conversation JSON files in a directory."""
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("conv_*.json"))
    for path in files:
        yield from ingest_conversation_file(path)


def ingest_bulk_conversations(zip_or_json_path: str | Path) -> Iterator[Segment]:
    """Ingest the bulk conversations.json from a ChatGPT export."""
    path = Path(zip_or_json_path)

    if path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            with z.open("conversations.json") as f:
                conversations = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            conversations = json.load(f)

    file_hash = sha256(path.read_bytes()).hexdigest()

    for conv in conversations:
        conv_id = conv.get("id", conv.get("conversation_id", "unknown"))
        title = conv.get("title", "untitled")
        create_time = conv.get("create_time")
        timestamp = (
            datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat()
            if create_time else "unknown"
        )

        provenance = SourceProvenance(
            source_id=f"chat:{conv_id}",
            tier=Tier.T3,
            url=f"chatgpt-export/{title[:60]}",
            commit_or_version=timestamp,
            license="proprietary",
            acquired_at=datetime.now(timezone.utc).isoformat(),
            artifact_sha256=file_hash,
        )

        mapping = conv.get("mapping", {})
        offset = 0

        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg:
                continue

            role = msg.get("author", {}).get("role", "unknown")
            if role == "system":
                continue

            content = msg.get("content", {})
            parts = content.get("parts", [])
            text = ""
            for part in parts:
                if isinstance(part, str):
                    text += part

            text = text.strip()
            if not text:
                continue

            byte_start = offset
            byte_end = offset + len(text.encode("utf-8"))
            offset = byte_end

            yield Segment(
                source=provenance,
                byte_range=(byte_start, byte_end),
                text=text,
                segment_type=f"chat_{role}",
            )
