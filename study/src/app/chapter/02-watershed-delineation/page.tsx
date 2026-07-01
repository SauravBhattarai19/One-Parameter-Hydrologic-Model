import ChapterContent from '../../../content/02-watershed-delineation.mdx';
import Link from 'next/link';

export const metadata = {
  title: 'Ch.2 — Watershed Delineation | OPM Hydrology',
};

export default function Chapter02() {
  return (
    <article className="chapter-body">
      <div className="not-prose flex items-center gap-2 text-sm text-slate-400 mb-8">
        <Link href="/" className="hover:text-sky-600 transition-colors">Chapters</Link>
        <span>›</span>
        <span className="text-slate-600">Chapter 2</span>
      </div>

      <ChapterContent />

      <div className="not-prose mt-16 pt-8 border-t border-slate-200 flex justify-between">
        <Link
          href="/chapter/01-dem-flow-direction"
          className="flex items-center gap-2 rounded-xl border border-slate-200 hover:border-sky-300 text-slate-600 hover:text-sky-600 px-5 py-3 text-sm font-semibold transition-colors"
        >
          ← Chapter 1 — DEMs & Flow Direction
        </Link>
        <Link
          href="/chapter/03-rainfall-runoff"
          className="flex items-center gap-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white px-5 py-3 text-sm font-semibold transition-colors"
        >
          Chapter 3 — Rainfall–Runoff →
        </Link>
      </div>
    </article>
  );
}
