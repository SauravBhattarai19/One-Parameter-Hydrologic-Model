import ChapterContent from '../../../content/05-diffusive-wave.mdx';
import Link from 'next/link';

export const metadata = {
  title: 'Ch.5 — Diffusive-Wave Routing | OPM Hydrology',
};

export default function Chapter05() {
  return (
    <article className="chapter-body">
      <div className="not-prose flex items-center gap-2 text-sm text-slate-400 mb-8">
        <Link href="/" className="hover:text-sky-600 transition-colors">Chapters</Link>
        <span>›</span>
        <span className="text-slate-600">Chapter 5</span>
      </div>

      <ChapterContent />

      <div className="not-prose mt-16 pt-8 border-t border-slate-200 flex justify-between">
        <Link
          href="/chapter/04-kinematic-wave"
          className="flex items-center gap-2 rounded-xl border border-slate-200 hover:border-sky-300 text-slate-600 hover:text-sky-600 px-5 py-3 text-sm font-semibold transition-colors"
        >
          ← Chapter 4 — Kinematic-Wave Routing
        </Link>
        <div />
      </div>
    </article>
  );
}
