// The equations of the documentation: the generic mode of pymdownx.arithmatex
// writes them as `\( ... \)` and `\[ ... \]` in elements of the class
// `arithmatex`, which MathJax 3 typesets. The configuration follows the one of
// the Zensical documentation; `document$` emits on the first load and on every
// page the instant navigation swaps in, so each page is typeset.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
