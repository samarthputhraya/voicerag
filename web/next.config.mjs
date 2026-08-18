/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // A fully static export: `next build` writes `out/` and nothing needs a Node
  // server at runtime. That is what lets the FastAPI container serve the
  // frontend from its own origin, which in turn is what makes the microphone
  // work with no configuration -- CORS does not apply to same-origin fetches,
  // and `stt_relay._origin_allowed` sees an Origin it already trusts.
  //
  // Nothing is lost here: every page in this app is a client component driven
  // entirely by fetches to the API, so there was never any server rendering to
  // give up. The one constraint to remember is that `output: "export"` runs the
  // page through a Node prerender at build time, so module-scope code must not
  // touch `window` -- see the lazy resolution in lib/sarvam.ts and the lazy
  // ASSET_BASE in app/page.tsx.
  output: "export",
  // No Next image server exists in an export, so any use of next/image must
  // skip optimisation. Set explicitly rather than left to fail the build later.
  images: { unoptimized: true },
  // The VAD library ships ONNX/WASM assets that must be served as static files
  // rather than bundled, so they are copied into public/ by the postinstall step
  // documented in the README.
  webpack: (config) => {
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false };
    return config;
  },
};
export default nextConfig;
