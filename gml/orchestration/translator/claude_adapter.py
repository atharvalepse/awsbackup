from pathlib import Path

from orchestration.translator.base import TranslatorAdapter


_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "claude" / "v1.md.jinja"
)

# Claude responds well to XML-tagged context blocks (system-prompt convention).
_EMBEDDED = (
    "<context>\n"
    "{% if reason_from_scratch %}\n"
    "<note>No prior memory available; reason from scratch.</note>\n"
    "{% elif selected %}\n"
    "{% for rh in selected %}\n"
    "<memory id=\"{{ rh.record.id }}\" source=\"{{ rh.record.source }}\""
    "{% if rh.record.entity %} entity=\"{{ rh.record.entity }}\""
    "{% if rh.record.attribute %} attribute=\"{{ rh.record.attribute }}\""
    "{% endif %}{% endif %}>\n"
    "{{ rh.record.content }}\n"
    "</memory>\n"
    "{% endfor %}\n"
    "{% else %}\n"
    "<note>No relevant context retrieved.</note>\n"
    "{% endif %}\n"
    "</context>\n"
    "{% if reasoning_content %}\n"
    "\n"
    "<sam_reasoning>\n"
    "{{ reasoning_content }}\n"
    "</sam_reasoning>\n"
    "{% endif %}\n"
    "\n"
    "<provenance items=\"{{ total_items }}\" source=\"gml-memory\"/>\n"
    "\n"
    "User query follows below.\n"
)


class ClaudeAdapter(TranslatorAdapter):
    template_path = _DEFAULT_TEMPLATE_PATH
    embedded_template_source = _EMBEDDED

    def target_family_name(self) -> str:
        return "claude"
