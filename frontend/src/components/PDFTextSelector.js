// Small utility: exposes a hook-like capture function for selected text in PDFs
export const captureSelectedText = () => {
  const selected = window.getSelection()
  if (!selected) return '';
  return selected.toString().trim();
};

export default function PDFTextSelector() {
  return null;
}
