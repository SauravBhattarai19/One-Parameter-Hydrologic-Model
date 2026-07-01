import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

export default {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            '--tw-prose-body': '#1e293b',
            '--tw-prose-headings': '#0f172a',
            '--tw-prose-links': '#0369a1',
            maxWidth: '72ch',
            h1: { fontWeight: '800', letterSpacing: '-0.03em' },
            h2: { fontWeight: '700', borderBottom: '2px solid #e2e8f0', paddingBottom: '0.3em' },
            table: { fontSize: '0.875rem' },
          },
        },
      },
    },
  },
  plugins: [typography],
} satisfies Config;
