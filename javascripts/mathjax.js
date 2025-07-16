/* 让 MathJax 在 MkDocs 中正确识别 $...$ 与 $$...$$ */
window.MathJax = {
  tex: {
    inlineMath: [ ['$', '$'], ['\\(', '\\)'] ],
    displayMath: [ ['$$', '$$'], ['\\[', '\\]'] ],
    processEscapes: true,        // 允许 \$ 转义
    processEnvironments: true,   // 支持 \begin{equation}
  },
  options: {
    /* 只处理带 class="arithmatex" 的元素，避免整页扫描 */
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

/* mkdocs-material Instant Navigation 兼容 */
document$.subscribe(() => MathJax.typesetPromise());
