/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  images: { remotePatterns: [{ protocol: "https", hostname: "**" }] },
  async rewrites() { return [{ source: "/api/:path*", destination: process.env.NEXT_PUBLIC_API_URL + "/api/:path*" }]; },
};
module.exports = nextConfig;