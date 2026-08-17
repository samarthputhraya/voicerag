/**
 * Copy the VAD runtime assets into `public/`.
 *
 * `@ricky0123/vad-web` loads three things over HTTP at runtime rather than
 * through the bundler: the AudioWorklet processor, the Silero ONNX weights, and
 * the onnxruntime-web WASM binaries. They resolve against the site root, so they
 * have to exist as static files or the mic button hangs forever on "Loading
 * voice model..." with a 404 that never surfaces in the React tree.
 *
 * next.config.mjs already documented this step as existing. It did not. This is
 * that step, wired to `postinstall` so a fresh clone gets it without reading a
 * comment.
 */

import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, "..");
const publicDir = join(web, "public");
const nm = join(web, "node_modules");

// Only the WASM variants onnxruntime actually selects at runtime. The asyncify
// and jspi builds are alternate threading backends we never request; copying
// them would add ~38 MB to the deploy for nothing.
const assets = [
  ["@ricky0123/vad-web/dist/vad.worklet.bundle.min.js", "vad.worklet.bundle.min.js"],
  ["@ricky0123/vad-web/dist/silero_vad_v5.onnx", "silero_vad_v5.onnx"],
  ["@ricky0123/vad-web/dist/silero_vad_legacy.onnx", "silero_vad_legacy.onnx"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.wasm"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.jsep.wasm"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.mjs"],
];

mkdirSync(publicDir, { recursive: true });

let copied = 0;
const missing = [];
for (const [from, to] of assets) {
  const src = join(nm, from);
  if (!existsSync(src)) {
    missing.push(from);
    continue;
  }
  copyFileSync(src, join(publicDir, to));
  copied += 1;
}

console.log(`[vad-assets] copied ${copied}/${assets.length} into public/`);
if (missing.length) {
  // Not fatal: a missing optional WASM variant should not fail `npm install`.
  // A missing worklet or model will fail loudly at runtime instead, which is
  // where it is actually diagnosable.
  console.warn(`[vad-assets] not found (skipped): ${missing.join(", ")}`);
}
