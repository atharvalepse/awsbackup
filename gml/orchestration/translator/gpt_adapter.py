from pathlib import Path

from orchestration.translator.base import TranslatorAdapter


_DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "gpt" / "v1.md.jinja"

_EMBEDDED = (
    "## Context\n"
    "\n"
    "The following relevant information was retrieved from GML memory:\n"
    "{% if reason_from_scratch %}\n"
    "(no prior memory available; reason from scratch)\n"
    "{% elif selected %}\n"
    "{% for rh in selected %}\n"
    "- **{{ rh.record.source }}**"
    "{% if rh.record.entity %} ({{ rh.record.entity }}"
    "{% if rh.record.attribute %}: {{ rh.record.attribute }}"
    "{% endif %}){% endif %}: {{ rh.record.content }}\n"
    "{% endfor %}\n"
    "{% else %}\n"
    "(no relevant context retrieved)\n"
    "{% endif %}\n"
    "{% if reasoning_content %}\n"
    "\n"
    "### SAM Reasoning\n"
    "\n"
    "{{ reasoning_content }}\n"
    "{% endif %}\n"
    "\n"
    "---\n"
    "*Retrieved {{ total_items }} item(s) from GML memory.*\n"
    "\n"
    "User query follows below.\n"
)


class GPTAdapter(TranslatorAdapter):
    template_path = _DEFAULT_TEMPLATE_PATH
    embedded_template_source = _EMBEDDED

    def target_family_name(self) -> str:
        return "gpt"
