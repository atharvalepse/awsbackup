from pathlib import Path

from orchestration.translator.base import TranslatorAdapter


_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "llama" / "v1.md.jinja"
)

# Llama instruct-tuned models do well with explicit role-marked sections.
_EMBEDDED = (
    "### Context (from GML memory)\n"
    "\n"
    "{% if reason_from_scratch %}\n"
    "_No prior memory available. Reason from scratch._\n"
    "{% elif selected %}\n"
    "{% for rh in selected %}\n"
    "[{{ loop.index }}] {{ rh.record.source }}"
    "{% if rh.record.entity %} ({{ rh.record.entity }}"
    "{% if rh.record.attribute %}: {{ rh.record.attribute }}"
    "{% endif %}){% endif %}\n"
    "{{ rh.record.content }}\n"
    "\n"
    "{% endfor %}\n"
    "{% else %}\n"
    "_No relevant context retrieved._\n"
    "{% endif %}\n"
    "{% if reasoning_content %}\n"
    "\n"
    "### SAM Reasoning\n"
    "{{ reasoning_content }}\n"
    "{% endif %}\n"
    "\n"
    "### Provenance\n"
    "{{ total_items }} item(s) retrieved from GML memory.\n"
    "\n"
    "### User Query\n"
)


class LlamaAdapter(TranslatorAdapter):
    template_path = _DEFAULT_TEMPLATE_PATH
    embedded_template_source = _EMBEDDED

    def target_family_name(self) -> str:
        return "llama"
