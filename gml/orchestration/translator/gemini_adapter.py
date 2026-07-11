from pathlib import Path

from orchestration.translator.base import TranslatorAdapter


_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "gemini" / "v1.md.jinja"
)

_EMBEDDED = (
    "# Retrieved Context\n"
    "\n"
    "The following information was retrieved from GML memory and may be "
    "relevant to the user's query.\n"
    "\n"
    "{% if reason_from_scratch %}\n"
    "*(no prior memory available; the assistant should reason from scratch)*\n"
    "{% elif selected %}\n"
    "## Memory Items\n"
    "\n"
    "{% for rh in selected %}\n"
    "### {{ loop.index }}. {{ rh.record.source }}\n"
    "{% if rh.record.entity %}**Entity:** {{ rh.record.entity }}"
    "{% if rh.record.attribute %} / {{ rh.record.attribute }}{% endif %}\n"
    "{% endif %}\n"
    "\n"
    "{{ rh.record.content }}\n"
    "\n"
    "{% endfor %}\n"
    "{% else %}\n"
    "*(no relevant context retrieved)*\n"
    "{% endif %}\n"
    "{% if reasoning_content %}\n"
    "\n"
    "## SAM Reasoning\n"
    "\n"
    "{{ reasoning_content }}\n"
    "{% endif %}\n"
    "\n"
    "---\n"
    "\n"
    "**Provenance:** {{ total_items }} item(s) retrieved from GML memory.\n"
    "\n"
    "---\n"
    "\n"
    "User query follows below.\n"
)


class GeminiAdapter(TranslatorAdapter):
    template_path = _DEFAULT_TEMPLATE_PATH
    embedded_template_source = _EMBEDDED

    def target_family_name(self) -> str:
        return "gemini"
