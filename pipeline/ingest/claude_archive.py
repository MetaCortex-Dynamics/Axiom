"""
Ingest Claude/Anthropic conversation archives into the governed pipeline.

Reads the bulk conversations.json from Claude data export.
Format: conversations[].chat_messages[].{sender, text, created_at}
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterator

from pipeline.stages.s1_segment import Segment
from pipeline.types import SourceProvenance, Tier


def ingest_claude_archive(zip_path: str | Path) -> Iterator[Segment]:
    """Ingest the Claude data export ZIP."""
    zip_path = Path(zip_path)
    file_hash = sha256(zip_path.read_bytes()).hexdigest()

    with zipfile.ZipFile(zip_path) as z:
        with z.open("conversations.json") as f:
            conversations = json.load(f)

    for conv in conversations:
        conv_id = conv.get("uuid", "unknown")
        name = conv.get("name", "untitled")
        created = conv.get("created_at", "")

        provenance = SourceProvenance(
            source_id=f"claude:{conv_id}",
            tier=Tier.T3,
            url=f"claude-export/{name[:60]}",
            commit_or_version=created,
            license="proprietary",
            acquired_at=datetime.now(timezone.utc).isoformat(),
            artifact_sha256=file_hash,
        )

        messages = conv.get("chat_messages", [])
        offset = 0

        for msg in messages:
            sender = msg.get("sender", "unknown")
            if sender == "system":
                continue

            # Claude format: text field OR content field (content may be list)
            text = msg.get("text", "")
            if not text:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                elif isinstance(content, str):
                    text = content

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
                segment_type=f"chat_{sender}",
            )
