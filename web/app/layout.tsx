import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VoiceRAG — grounded answers from speech",
  description:
    "Speak a question, get an answer grounded in MS MARCO with citations, guardrails, and a per-stage latency receipt.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
