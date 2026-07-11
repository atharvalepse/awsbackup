from pathlib import Path

from orchestration.translator.base import TranslatorAdapter


_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "deepseek" / "v1.md.jinja"
)

# DeepSeek R1 is a reasoning model: it produces <think>...</think> blocks
# internally, so context is delivered with explicit section markers and a
# clear instruction to think step-by-step.
_EMBEDDED = (
    "## Retrieved Context\n"
    "\n"
    "{% if reason_from_scratch %}\n"
    "No prior memory was retrieved; reason from scratch.\n"
    "{% elif selected %}\n"
    "{% for rh in selected %}\n"
    "**[{{ loop.index }}] {{ rh.record.source }}**"
    "{% if rh.record.entity %} — {{ rh.record.entity }}"
    "{% if rh.record.attribute %}/{{ rh.record.attribute }}{% endif %}{% endif %}\n"
    "{{ rh.record.content }}\n"
    "\n"
    "{% endfor %}\n"
    "{% else %}\n"
    "No relevant context retrieved.\n"
    "{% endif %}\n"
    "{% if reasoning_content %}\n"
    "\n"
    "## Prior Reasoning (from SAM)\n"
    "\n"
    "{{ reasoning_content }}\n"
    "{% endif %}\n"
    "\n"
    "## Provenance\n"
    "{{ total_items }} item(s) retrieved from GML memory.\n"
    "\n"
    "## User Query\n"
)


class DeepSeekAdapter(TranslatorAdapter):
    template_path = _DEFAULT_TEMPLATE_PATH
    embedded_template_source = _EMBEDDED

    def target_family_name(self) -> str:
        return "deepseek"
