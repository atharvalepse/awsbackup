"""Canonical persisted memory format: simplemem + sjson.

Public surface:

  * :class:`AAL`           — one record (simplemem + sjson + provenance)
  * :class:`AALBundle`     — multiple AALs from a single turn
  * :class:`AALConverter`  — turn raw input or LLM-extracted items into AAL

Importing pattern::

    from orchestration.aal import AAL, AALBundle, AALConverter
"""
from orchestration.aal.converter import AALConverter
from orchestration.aal.record import AAL, AALBundle

__all__ = ["AAL", "AALBundle", "AALConverter"]
