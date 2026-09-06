/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  root: "console",
  // relative base so the built dist can be served by `ao preview` as a static file
  base: "./",
  plugins: [react(), tailwindcss()],
  server: {
    // same-origin /api proxy to the local Python control API. changeOrigin
    // stays false so the Host header remains the console origin: the API's
    // access boundary requires a loopback Host and a same-origin request, and
    // the HttpOnly operator cookie is set on this origin.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    environment: "happy-dom",
    globals: true,
    setupFiles: ["./console/src/test/setup.ts"],
  },
});
