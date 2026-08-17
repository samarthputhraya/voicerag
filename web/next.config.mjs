/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The VAD library ships ONNX/WASM assets that must be served as static files
  // rather than bundled, so they are copied into public/ by the postinstall step
  // documented in the README.
  webpack: (config) => {
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false };
    return config;
  },
};
export default nextConfig;
