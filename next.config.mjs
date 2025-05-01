// @ts-check

import bundleAnalyzer from '@next/bundle-analyzer';

const withBundleAnalyzer = bundleAnalyzer({
  enabled: process.env.ANALYZE === 'true',
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['ipfs.io', 'localhost'],
  },
  webpack: (config) => {
    // Custom webpack configuration
    return config;
  },
};

export default withBundleAnalyzer(nextConfig); 