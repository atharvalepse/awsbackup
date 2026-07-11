"""SDP Stage 2 — SemanticExtractor.

The lightweight extraction engine the doc prescribes: regex, heuristics,
keyword extraction. No LLM. Produces a list of raw extractions (dicts)
that downstream stages will refine into SemanticUnits and AALMemory.

Each extraction is shaped:
  {"type": "fact" | "preference" | "config" | "decision" | "issue",
   "category": str,                  # technology / version / port / person / url / ...
   "value": str,                     # the actual value extracted
   "subject": str | None,            # the entity it's about, when detectable
   "attribute": str | None,          # the property it's about
   "span": str}                      # the surrounding clause (for provenance)
"""
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Patterns — kept readable. Add more as needed; this is the lightweight
# baseline the doc calls for, NOT an exhaustive NER system.
# ---------------------------------------------------------------------------


# Technology / framework / database / language vocab. Order matters — longer
# matches first so e.g. "PostgreSQL 16" beats a bare "Postgres".
_TECH_TERMS = [
    "PostgreSQL", "Postgres", "MySQL", "MariaDB", "SQLite", "MongoDB", "Redis",
    "Memcached", "Cassandra", "DynamoDB", "Elasticsearch", "OpenSearch",
    "FastAPI", "Flask", "Django", "Express", "NestJS", "Next.js", "Nuxt", "Remix",
    "Spring Boot", "Spring", "Rails", "Sinatra", "Phoenix", "Gin", "Echo",
    "React", "Vue", "Svelte", "Solid", "Angular", "Ember",
    "Kubernetes", "Docker", "ECS", "EKS", "GKE", "ArgoCD", "Helm",
    "GitHub Actions", "GitLab CI", "Jenkins", "CircleCI", "Buildkite",
    "Stripe", "Adyen", "Braintree", "Square", "PayPal", "Plaid", "Twilio",
    "Auth0", "Okta", "Clerk", "Cognito", "Firebase", "Supabase",
    "Honeycomb", "Datadog", "Grafana", "Prometheus", "Sentry", "New Relic",
    "OpenTelemetry", "Jaeger", "Zipkin",
    "Qdrant", "Pinecone", "Weaviate", "Milvus", "Chroma", "pgvector",
    "Kafka", "RabbitMQ", "NATS", "SQS", "PubSub",
    "S3", "GCS", "R2", "Azure Blob",
    "Upstash", "ElastiCache", "Lambda", "Cloud Run", "Fly.io", "Render",
    "Vercel", "Netlify", "Heroku", "AWS", "GCP", "Azure",
    "Go", "Rust", "Python", "TypeScript", "JavaScript", "Java", "Kotlin",
    "Ruby", "PHP", "C#", "C++", "Swift", "Scala", "Elixir",
    "Anthropic", "OpenAI", "Gemini", "Claude", "DeepSeek", "Qwen", "Llama",
    "Mistral", "Ollama", "FastEmbed", "BAAI/bge-small-en-v1.5",
    "Node.js", "Node", "Deno", "Bun", "GraphQL", "gRPC", "tRPC", "REST",
    "Terraform", "Ansible", "Pulumi", "Nginx", "Caddy", "Apache", "HAProxy",
    "Celery", "Sidekiq", "Temporal", "Airflow", "dbt", "Snowflake",
    "BigQuery", "Databricks", "Spark", "Hadoop", "Flink",
    "Tailwind", "Vite", "Webpack", "esbuild", "Turbopack", "Rollup",
    "Playwright", "Cypress", "Selenium", "Jest", "Vitest", "Pytest", "Mocha",
    "Prisma", "Drizzle", "SQLAlchemy", "TypeORM", "Sequelize", "Hibernate",
    "Hasura", "Apollo", "WebSocket", "Socket.IO",
    "Cloudflare", "Fastly", "Akamai", "DigitalOcean", "Hetzner", "Linode",
    "MinIO", "Backblaze", "Cohere", "HuggingFace", "LangChain", "LlamaIndex",
    "Pandas", "NumPy", "PyTorch", "TensorFlow", "scikit-learn", "JAX",
    "Tableau", "Looker", "Metabase", "Superset", "Plotly",
]
_TECH_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(_TECH_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b")
_PORT_RE = re.compile(r"\bport\s+(\d{2,5})\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b((?:https?://)?[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)\b", re.IGNORECASE)
_PERSON_RE = re.compile(
    r"\b([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)\s+"
    r"(is|runs|leads|owns|handles|manages|covers)\b"
)

# Verbs that flag explicit factual statements worth capturing.
_FACT_TRIGGER = re.compile(
    r"\b(use|uses|using|run|runs|running|host|hosts|hosted|deploy|deployed|deploying|"
    r"migrate|migrated|upgrade|upgraded|switch|switched|rolled out|"
    r"set up|configured|live[s]?\s+at|is on|now (?:on|runs)|"
    r"ship|ships|shipped|shipping|build|builds|built|building|"
    r"work[s]?\s+(?:at|for|on|with)|working\s+(?:at|for|on|with)|worked\s+(?:at|for|on|with)|"
    r"based\s+(?:in|on|at)|located\s+(?:in|at)|live[sd]?\s+in|living\s+in|"
    r"write[s]?|wrote|written|store[ds]?|stored|cache[ds]?|adopt(?:ed|s)?|"
    r"integrate[ds]?|choose|chose|standardiz|email|contact|call(?:ed)?|named|"
    r"handle[ds]?|leads?|owns?|manages?|covers?|is\s+(?:in|on|at)|"
    r"is\s+\w+|are\s+\w+|consists of|comprises)\b",
    re.IGNORECASE,
)
# Personal / identity facts — self-contained patterns that don't need a tech
# term to be worth keeping. These let SDP capture name/role/employer/location/
# email deterministically (no LLM), which the verb-triggered extractors miss.
_NAME_RE = re.compile(
    r"(?i:\b(?:my name(?:'s| is)|i am|i'm|call me|this is)\s+)"
    r"([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15}){0,2})\b"
)
_ROLE_RE = re.compile(
    r"\b(?:i'?m|i am|work as|working as)\s+(?:an?\s+)?"
    r"([\w/+\- ]*?(?:engineer|developer|designer|manager|scientist|analyst|"
    r"architect|lead|founder|consultant|researcher|student|teacher|writer|"
    r"devops|sre|administrator|programmer|intern|cto|ceo|cfo|pm))\b",
    re.IGNORECASE,
)
_EMPLOYER_RE = re.compile(
    r"\b(?:work(?:s|ed|ing)?\s+(?:at|for)|employed\s+(?:at|by))\s+"
    r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})\b"
)
_LOCATION_RE = re.compile(
    r"(?i:\b(?:live[sd]?|living|based|located|reside[sd]?)\s+in\s+)"
    r"([A-Z][\w.\-]*(?:[ ]+[A-Z][\w.\-]*){0,2})\b"
)
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
# Decisions / policies.
_DECISION_TRIGGER = re.compile(
    r"\b(decided|chose|going with|will use|won't use|standardize on|"
    r"prohibit(?:ed)?|not allowed|require[ds]?|must|never)\b",
    re.IGNORECASE,
)
# Issues / known bugs.
_ISSUE_TRIGGER = re.compile(
    r"\b(bug|issue|problem|incident|outage|broken|fail(?:ed|ing|ure)?|crash)\b",
    re.IGNORECASE,
)
# Preferences ("we prefer", "we like", "user wants").
_PREFERENCE_TRIGGER = re.compile(
    r"\b(prefer(?:s|red)?|like(?:s|d)?|favorite|wants?|want(?:ed)?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------


@dataclass
class _Extraction:
    type: str
    category: str
    value: str
    subject: str | None
    attribute: str | None
    span: str


def _clauses(text: str) -> list[str]:
    """Split on sentence boundaries + obvious clause boundaries.

    Cheap and good enough — we don't need linguistic correctness here, just
    units small enough that one extraction per clause is sensible.
    """
    parts = re.split(r"(?<=[.!?])\s+|;\s+| — | -- ", text)
    return [p.strip() for p in parts if p and p.strip()]


def _detect_type(span: str) -> str:
    if _ISSUE_TRIGGER.search(span):
        return "issue"
    if _DECISION_TRIGGER.search(span):
        return "decision"
    if _PREFERENCE_TRIGGER.search(span):
        return "preference"
    return "fact"


class SemanticExtractor:
    """Regex-driven semantic extraction. Lightweight, deterministic, fast."""

    def extract(self, text: str) -> list[dict]:
        if not text:
            return []

        extractions: list[_Extraction] = []
        for span in _clauses(text):
            # Identity/personal facts (name/role/employer/location/email) are
            # self-contained — capture them regardless of a fact-trigger verb,
            # since "my name is Sam" has no tech term to anchor on.
            self._extract_identity(span, extractions)
            if not _FACT_TRIGGER.search(span) and not _DECISION_TRIGGER.search(span) and \
               not _PREFERENCE_TRIGGER.search(span) and not _ISSUE_TRIGGER.search(span):
                continue
            ext_type = _detect_type(span)
            self._extract_techs(span, ext_type, extractions)
            self._extract_versions(span, ext_type, extractions)
            self._extract_ports(span, ext_type, extractions)
            self._extract_urls(span, ext_type, extractions)
            self._extract_people(span, ext_type, extractions)

        # Deduplicate on (category, value, subject) within one extract() call.
        seen = set()
        out: list[dict] = []
        for e in extractions:
            key = (e.category, e.value.lower(), (e.subject or "").lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "type": e.type,
                "category": e.category,
                "value": e.value,
                "subject": e.subject,
                "attribute": e.attribute,
                "span": e.span,
            })
        return out

    # --- per-category extractors --------------------------------------

    def _extract_techs(self, span: str, ext_type: str, out: list[_Extraction]) -> None:
        for m in _TECH_RE.finditer(span):
            value = m.group(1)
            # Normalize case to the canonical term we keep in _TECH_TERMS
            canonical = next(
                (t for t in _TECH_TERMS if t.lower() == value.lower()),
                value,
            )
            out.append(_Extraction(
                type=ext_type, category="technology", value=canonical,
                subject=None, attribute="stack",
                span=span,
            ))

    def _extract_versions(self, span: str, ext_type: str, out: list[_Extraction]) -> None:
        # Pair version numbers with the closest preceding tech term in the same span.
        techs = list(_TECH_RE.finditer(span))
        for vm in _VERSION_RE.finditer(span):
            version = vm.group(1)
            # Find closest preceding tech in the span (by character index)
            preceding = [t for t in techs if t.start() < vm.start()]
            subject = preceding[-1].group(1) if preceding else None
            out.append(_Extraction(
                type=ext_type, category="version", value=version,
                subject=subject, attribute="version",
                span=span,
            ))

    def _extract_ports(self, span: str, ext_type: str, out: list[_Extraction]) -> None:
        for m in _PORT_RE.finditer(span):
            port = m.group(1)
            out.append(_Extraction(
                type=ext_type, category="port", value=port,
                subject=None, attribute="port",
                span=span,
            ))

    def _extract_urls(self, span: str, ext_type: str, out: list[_Extraction]) -> None:
        for m in _URL_RE.finditer(span):
            url = m.group(1)
            # Filter out things matched as URL that are actually filenames or version tags
            if "." not in url or url.endswith((".js", ".py", ".md", ".txt", ".json")):
                continue
            if re.match(r"^\d+\.\d+", url):  # version-like
                continue
            out.append(_Extraction(
                type=ext_type, category="url", value=url,
                subject=None, attribute="location",
                span=span,
            ))

    def _extract_identity(self, span: str, out: list[_Extraction]) -> None:
        """Personal facts: name / role / employer / location / email."""
        for m in _NAME_RE.finditer(span):
            out.append(_Extraction(
                type="fact", category="name", value=m.group(1).strip(),
                subject=None, attribute="name", span=span))
        for m in _ROLE_RE.finditer(span):
            out.append(_Extraction(
                type="fact", category="role", value=m.group(1).strip(),
                subject=None, attribute="role", span=span))
        for m in _EMPLOYER_RE.finditer(span):
            out.append(_Extraction(
                type="fact", category="employer", value=m.group(1).strip(),
                subject=None, attribute="employer", span=span))
        for m in _LOCATION_RE.finditer(span):
            out.append(_Extraction(
                type="fact", category="location", value=m.group(1).strip(),
                subject=None, attribute="location", span=span))
        for m in _EMAIL_RE.finditer(span):
            out.append(_Extraction(
                type="fact", category="email", value=m.group(1).strip(),
                subject=None, attribute="email", span=span))

    def _extract_people(self, span: str, ext_type: str, out: list[_Extraction]) -> None:
        for m in _PERSON_RE.finditer(span):
            name, verb = m.group(1), m.group(2).lower()
            attr = {
                "leads": "lead", "runs": "runs", "owns": "owner",
                "handles": "handler", "manages": "manager", "covers": "cover",
                "is": "role",
            }.get(verb, "role")
            out.append(_Extraction(
                type=ext_type, category="person", value=name,
                subject=None, attribute=attr,
                span=span,
            ))
