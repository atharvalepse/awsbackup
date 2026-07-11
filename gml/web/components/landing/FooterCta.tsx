import Link from "next/link";

export function FooterCta() {
  return (
    <Link
      href="/app"
      className="group block border-t border-border bg-bg-1 transition-shadow duration-[180ms] ease-out hover:shadow-glow"
    >
      <div className="mx-auto flex max-w-5xl items-center justify-between px-8 py-12 sm:px-16">
        <span className="text-2xl font-medium text-text-0">
          Open the graph
          <span className="ml-2 inline-block text-accent transition-transform duration-[180ms] ease-out group-hover:translate-x-1">
            →
          </span>
        </span>
        <span className="font-mono text-xs text-text-2">/app</span>
      </div>
    </Link>
  );
}
