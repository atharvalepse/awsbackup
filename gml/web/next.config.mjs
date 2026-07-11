import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // Force EVERY bare `import 'three'` (react-three-fiber/drei AND
    // react-force-graph-3d's three-forcegraph) to resolve to ONE module
    // instance. Without this, webpack bundles three into multiple chunks →
    // "Multiple instances of Three.js" → the force-graph render loop crashes
    // ("Cannot read properties of undefined (reading 'tick')") → black canvas.
    // Exact-match ($) so subpath imports like `three/examples/*` still resolve.
    config.resolve.alias = {
      ...config.resolve.alias,
      three$: path.resolve(__dirname, "node_modules/three/build/three.module.js"),
    };
    return config;
  },
};

export default nextConfig;
