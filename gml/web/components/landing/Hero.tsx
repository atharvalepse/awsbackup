"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";

// three.js stays out of the LCP path — loaded only after first paint.
const HeroScene = dynamic(() => import("./HeroScene"), {
  ssr: false,
  loading: () => null,
});

export function Hero() {
  return (
    <section className="relative grid min-h-screen grid-cols-1 lg:grid-cols-2">
      {/* left — typography. CSS entrance (animate-rise) so text is never
          JS-dependent for visibility; respects prefers-reduced-motion globally. */}
      <div className="z-10 flex flex-col justify-center px-8 py-24 sm:px-16">
        <p
          className="animate-rise font-mono text-xs tracking-[0.12em] text-text-2"
          style={{ animationDelay: "0ms" }}
        >
          GML / Memory Layer for LLMs
        </p>

        <h1
          className="animate-rise mt-5 font-display text-5xl font-medium leading-[1.05] tracking-tight text-text-0 sm:text-6xl lg:text-[72px]"
          style={{ animationDelay: "60ms" }}
        >
          Memories that compose<span className="text-accent">.</span>
        </h1>

        <p
          className="animate-rise mt-6 max-w-[480px] text-lg leading-relaxed text-text-1"
          style={{ animationDelay: "130ms" }}
        >
          GML gives every LLM a graph of what it knows, who said it, and how sure
          it is.
        </p>

        <div
          className="animate-rise mt-9 flex flex-wrap items-center gap-3"
          style={{ animationDelay: "200ms" }}
        >
          <Link href="/app">
            <Button size="md">Enter the graph →</Button>
          </Link>
          <Link href="/app/trace">
            <Button variant="ghost" size="md">
              Read the trace
            </Button>
          </Link>
        </div>
      </div>

      {/* right — three.js scene */}
      <div className="absolute inset-0 lg:relative lg:inset-auto">
        <div className="h-full w-full opacity-60 lg:opacity-100">
          <HeroScene />
        </div>
      </div>

      {/* fade the scene behind the text on small screens for legibility */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-bg-0 via-bg-0/70 to-transparent lg:hidden" />
    </section>
  );
}
