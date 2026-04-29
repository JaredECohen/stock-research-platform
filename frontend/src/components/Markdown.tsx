import React, { useMemo } from "react";

// Tiny safe-ish markdown renderer for chat answers.
// Supports: headings (#), bold (**), italic (_), bullets (-), and paragraphs.

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]!));
}

function renderInline(line: string): string {
  let s = escapeHtml(line);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/_(.+?)_/g, "<em>$1</em>");
  s = s.replace(/`(.+?)`/g, '<code class="bg-ink-900 px-1 rounded">$1</code>');
  return s;
}

export function Markdown({ text }: { text: string }) {
  const html = useMemo(() => {
    const lines = text.split(/\r?\n/);
    const out: string[] = [];
    let inList = false;
    for (const raw of lines) {
      const line = raw.trim();
      if (line.startsWith("### ")) {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        out.push(`<h3 class="text-base font-semibold mt-4 mb-1.5">${renderInline(line.slice(4))}</h3>`);
      } else if (line.startsWith("## ")) {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        out.push(`<h2 class="text-lg font-semibold mt-4 mb-2">${renderInline(line.slice(3))}</h2>`);
      } else if (line.startsWith("- ")) {
        if (!inList) {
          out.push('<ul class="list-disc pl-5 space-y-1 my-2">');
          inList = true;
        }
        out.push(`<li>${renderInline(line.slice(2))}</li>`);
      } else if (line === "") {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        out.push("");
      } else {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        out.push(`<p class="leading-relaxed my-1.5">${renderInline(line)}</p>`);
      }
    }
    if (inList) out.push("</ul>");
    return out.join("\n");
  }, [text]);

  return <div className="prose-invert text-sm text-slate-200" dangerouslySetInnerHTML={{ __html: html }} />;
}
