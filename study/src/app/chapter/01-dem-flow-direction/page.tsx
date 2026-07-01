import ChapterContent from '../../../content/01-dem-and-flow-direction.mdx';
import Link from 'next/link';

export const metadata = {
  title: 'Ch.1 — Digital Elevation Models & Flow Direction | OPM Hydrology',
};

export default function Chapter01() {
  return (
    <article className="chapter-body">
      {/* Chapter breadcrumb */}
      <div className="not-prose flex items-center gap-2 text-sm text-slate-400 mb-8">
        <Link href="/" className="hover:text-sky-600 transition-colors">Chapters</Link>
        <span>›</span>
        <span className="text-slate-600">Chapter 1</span>
      </div>

      <ChapterContent />

      {/* Chapter navigation */}
      <div className="not-prose mt-16 pt-8 border-t border-slate-200 flex justify-between">
        <div /> {/* no previous chapter */}
        <Link
          href="/chapter/02-watershed-delineation"
          className="flex items-center gap-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white px-5 py-3 text-sm font-semibold transition-colors"
        >
          Chapter 2 — Watershed Delineation →
        </Link>
      </div>
    </article>
  );
}
