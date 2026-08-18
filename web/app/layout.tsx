import type { Metadata } from "next";
import { Imbue, Victor_Mono } from "next/font/google";
import "./globals.css";

/**
 * The organisers' two faces, loaded the way hhgoa.com itself loads them: as CSS
 * custom properties, so every rule in globals.css can name a *role* rather than
 * a family. Their stylesheet exposes exactly `--font-imbue` and
 * `--font-victor-mono`; matching those names keeps this file readable next to
 * theirs.
 *
 * Both are variable fonts, so no weight list is declared — the full axis is
 * fetched once and `font-weight` interpolates freely. `display: swap` because a
 * demo that renders invisible text for the first 3 s of a judge's attention has
 * already lost, and both faces have close metric fallbacks.
 */
const imbue = Imbue({
  subsets: ["latin"],
  variable: "--font-imbue",
  display: "swap",
});

const victorMono = Victor_Mono({
  subsets: ["latin"],
  variable: "--font-victor-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VoiceRAG — grounded answers from speech",
  description:
    "Speak a question, get an answer grounded in MS MARCO with citations, guardrails, and a per-stage latency receipt.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${imbue.variable} ${victorMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
