"""Test-only delivery fixture for the pinned Flower 1.22 Message contract.

A live SuperLink assigns message IDs. Native in-memory tests have no SuperLink,
so construct a new received Message using Flower's INTERNAL metadata constructor.
This compatibility entry is pinned to 1.22 and is never bundled into a Flower app.
Do not add setters, write private metadata, or manufacture IDs in production.
"""
from __future__ import annotations

import copy
import uuid


def received_message(message, *, message_id: str | None = None):
    """Copy an instruction as a delivered test message, without mutating it.

    Preserve routing, run, group, timestamps, TTL and any existing ID. Explicit IDs
    support repeat/duplicate rejection tests. Content is deep-copied to model a
    transport boundary, rather than accidentally sharing mutable records.
    """
    from flwr.common import Message, Metadata

    if not isinstance(message, Message):
        raise TypeError('The fixture requires a native Flower Message')
    if message.has_error() or not message.has_content():
        raise ValueError('The fixture accepts content-bearing instructions only')
    if message.metadata.reply_to_message_id:
        raise ValueError('The fixture must not convert a reply into an instruction')
    if message_id is not None and (type(message_id) is not str or not message_id.strip()):
        raise ValueError('Explicit test message ID must be a nonempty string')

    source = message.metadata
    existing_id = source.message_id
    if type(existing_id) is not str:
        raise TypeError('Unexpected message ID type')
    if existing_id and not existing_id.strip():
        raise ValueError('Existing message ID must not contain only whitespace')
    if existing_id and message_id is not None and message_id != existing_id:
        raise ValueError('An existing request ID must not be overwritten')
    chosen_id = existing_id or message_id or uuid.uuid4().hex
    metadata = Metadata(
        run_id=source.run_id,
        message_id=chosen_id,
        src_node_id=source.src_node_id,
        dst_node_id=source.dst_node_id,
        reply_to_message_id=source.reply_to_message_id,
        group_id=source.group_id,
        created_at=source.created_at,
        ttl=source.ttl,
        message_type=source.message_type,
    )
    # Internal constructor verified in flwr/common/message.py at v1.22.0.
    received = Message(metadata=metadata, content=copy.deepcopy(message.content))
    received.metadata.delivered_at = source.delivered_at
    return received
