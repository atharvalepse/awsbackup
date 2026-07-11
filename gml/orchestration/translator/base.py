"""TranslatorAdapter ABC + shared Jinja-template loader.

Per-target rendering lives in subclasses. The base class handles the
file → embedded-fallback → error tiered template loading so each adapter
only needs to declare its template path and embedded template string.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import jinja2

from orchestration.errors import TranslatorError
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline.contracts import AssembledContext


slog = StructuredLogger("translator")


def _build_env(loader: jinja2.BaseLoader) -> jinja2.Environment:
    return jinja2.Environment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=True,
    )


class TranslatorAdapter(ABC):
    """Strategy: render an AssembledContext as a string the target AI expects.

    Subclasses provide ``template_path`` and ``embedded_template_source``.
    All Jinja loading + fallback logic is inherited from this base.
    """

    #: Absolute path to the on-disk template file (per-adapter default).
    template_path: Path

    #: Embedded fallback template used when the file is unavailable.
    embedded_template_source: str

    def __init__(self, template_path: str | Path | None = None) -> None:
        if template_path is not None:
            self.template_path = Path(template_path)
        self._template: jinja2.Template | None = None

    def _get_template(self) -> jinja2.Template:
        if self._template is not None:
            return self._template

        try:
            env = _build_env(jinja2.FileSystemLoader(str(self.template_path.parent)))
            self._template = env.get_template(self.template_path.name)
            return self._template
        except (jinja2.TemplateError, OSError) as file_exc:
            slog.warning(
                event="template_file_unavailable_using_embedded_fallback",
                template_path=str(self.template_path),
                error_type=type(file_exc).__name__,
                error=str(file_exc),
                degraded_mode=True,
            )

        try:
            env = _build_env(jinja2.BaseLoader())
            self._template = env.from_string(self.embedded_template_source)
            return self._template
        except jinja2.TemplateError as embedded_exc:
            raise TranslatorError(
                f"{type(self).__name__}: file template at {self.template_path!r} "
                f"failed AND embedded fallback failed to parse: {embedded_exc}"
            ) from embedded_exc

    @abstractmethod
    def target_family_name(self) -> str:
        """Short identifier for logging/metrics (e.g. ``'claude'``, ``'gpt'``)."""
        ...

    def render(self, context: AssembledContext) -> str:
        template = self._get_template()
        return template.render(
            selected=context.selected,
            query=context.query.text,
            improved_query=context.improved_query,
            reasoning_content=context.reasoning_content,
            reason_from_scratch=context.metadata.get("reason_from_scratch", False),
            notes=context.metadata.get("notes", []),
            total_items=len(context.selected),
        )

    def empty_template(self) -> str:
        """Render with no memories — used by Pipeline for budget overhead estimation."""
        return self._get_template().render(
            selected=[],
            query="",
            improved_query=None,
            reasoning_content=None,
            reason_from_scratch=False,
            notes=[],
            total_items=0,
        )
