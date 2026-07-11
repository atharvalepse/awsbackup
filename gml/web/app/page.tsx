import { Hero } from "@/components/landing/Hero";
import { HowItComposes } from "@/components/landing/HowItComposes";
import { LiveTrace } from "@/components/landing/LiveTrace";
import { FooterCta } from "@/components/landing/FooterCta";

export default function Home() {
  return (
    <main className="bg-bg-0">
      <Hero />
      <HowItComposes />
      <LiveTrace />
      <FooterCta />
    </main>
  );
}
