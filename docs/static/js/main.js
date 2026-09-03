/* MetaCaster project page — interactions. */
(function () {
  "use strict";

  const $  = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  /* ------------------------------------------------------------ language */
  /* Static copy carries its Chinese in data-zh; the strings below are the
     ones the tables build at runtime. */
  const ZH = {
    "Ours": "本文",
    "Generation models": "生成模型",
    "Augmentation": "数据增强",
    "References": "参考",
    "Dataset": "数据集",
    "Wins of 30": "30 组中最优",
    "In-domain corpus": "域内语料",
    "Out-of-domain corpus": "域外语料",
    "Out-of-domain": "域外",
    "Variant": "变体",
    "Overall": "总体",
    "Generation objective": "生成目标",
    "Textual context": "文本上下文",
    "LLM backbone": "大模型骨干",
    "Loss → MMD": "目标 → MMD",
    "Loss → Wasserstein": "目标 → Wasserstein",
    "Remove context 𝖢": "移除上下文 𝖢",
    "MetaCaster": "MetaCaster",
    "All": "全部",
    "Copy": "复制",
    "Copied": "已复制",
  };

  let lang = document.documentElement.getAttribute("data-lang") || "en";
  const tr = (s) => (lang === "zh" && ZH[s] ? ZH[s] : s);

  function applyLang(next) {
    lang = next;
    const root = document.documentElement;
    root.setAttribute("data-lang", lang);
    root.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
    localStorage.setItem("mc-lang", lang);

    $$("[data-zh]").forEach((n) => {
      if (n.dataset.en === undefined) n.dataset.en = n.innerHTML;
      n.innerHTML = lang === "zh" ? n.dataset.zh : n.dataset.en;
    });

    renderMain();
    renderAblation();
    renderLtlib();
  }

  $("#lang-toggle").addEventListener("click", () => applyLang(lang === "zh" ? "en" : "zh"));

  /* --------------------------------------------------------------- theme */
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("mc-theme", next);
  });

  /* ----------------------------------------------------------------- nav */
  const nav = $("#nav");
  const onScroll = () => nav.setAttribute("data-stuck", String(window.scrollY > 8));
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* --------------------------------------------------------------- menu */
  const menuBtn = $("#menu-toggle");
  const setMenu = (open) => {
    nav.setAttribute("data-menu", open ? "open" : "closed");
    menuBtn.setAttribute("aria-expanded", String(open));
  };
  menuBtn.addEventListener("click", () =>
    setMenu(nav.getAttribute("data-menu") !== "open"));
  $$(".nav-links a").forEach((a) => a.addEventListener("click", () => setMenu(false)));

  /* -------------------------------------------------------- copy bibtex */
  const copyBtn = $("#copy-bib");
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText($("#bibtex").textContent).then(() => {
      copyBtn.textContent = tr("Copied");
      copyBtn.dataset.done = "true";
      setTimeout(() => { copyBtn.textContent = tr("Copy"); delete copyBtn.dataset.done; }, 1800);
    });
  });

  /* --------------------------------------------------------------- utils */
  /* Best and second-best within a set of values; ties share a rank, which is
     how the paper marks its tables. */
  function rankMarks(values) {
    const sorted = [...new Set(values.filter(Number.isFinite))].sort((a, b) => a - b);
    return values.map((v) => (v === sorted[0] ? "best" : v === sorted[1] ? "second" : ""));
  }

  const fmt = (v) => (v >= 10 ? v.toFixed(2) : v.toFixed(3));

  /* A vertical rule between column groups, matching the paper's layout. */
  const GROUP_EDGES = new Set([1, 6]);   // TimeDP opens one group, Repeat the next

  function bindChoices(id, key, apply) {
    $$(`#${id} button`).forEach((b) => {
      b.addEventListener("click", () => {
        $$(`#${id} button`).forEach((x) => x.removeAttribute("aria-pressed"));
        b.setAttribute("aria-pressed", "true");
        apply(b.dataset[key]);
      });
    });
  }

  /* ----------------------------------------------------------- main table */
  const mainTable = $("#main-table");
  let curK = 30, curCorpus = "all";

  function buildMainHead() {
    const head = mainTable.tHead;
    head.innerHTML = "";

    const g = el("tr", "grp");
    g.appendChild(el("th", "name", ""));
    g.appendChild(el("th", "", tr("Ours")));
    const gm = el("th", "sep", tr("Generation models")); gm.colSpan = 5; g.appendChild(gm);
    const ga = el("th", "sep", tr("Augmentation"));      ga.colSpan = 4; g.appendChild(ga);
    const gr = el("th", "sep", tr("References"));        gr.colSpan = 2; g.appendChild(gr);
    head.appendChild(g);

    const h = el("tr");
    h.appendChild(el("th", "name", tr("Dataset")));
    METHODS.forEach((m, i) => h.appendChild(el("th", GROUP_EDGES.has(i) ? "sep" : "", m.label)));
    REFERENCES.forEach((r, i) => h.appendChild(el("th", i === 0 ? "sep" : "", r.label)));
    head.appendChild(h);
  }

  function dataRow(row) {
    const tr_ = el("tr");
    tr_.appendChild(el("td", "name", row[0]));

    const vals = row.slice(1, 11);
    const marks = rankMarks(vals);
    vals.forEach((v, i) => {
      const cls = [marks[i], i === 0 ? "ours" : "", GROUP_EDGES.has(i) ? "sep" : ""];
      tr_.appendChild(el("td", cls.filter(Boolean).join(" "), fmt(v)));
    });

    tr_.appendChild(el("td", "ref sep", fmt(row[11])));
    tr_.appendChild(el("td", "ref", fmt(row[12])));
    return tr_;
  }

  function renderMain() {
    buildMainHead();
    const body = mainTable.tBodies[0] || mainTable.createTBody();
    body.innerHTML = "";

    const block = MAIN_RESULTS[curK];
    const corpora = curCorpus === "all" ? ["IND", "OOD"] : [curCorpus];

    corpora.forEach((c) => {
      const label = tr(c === "IND" ? "In-domain corpus" : "Out-of-domain corpus");
      const sec = el("tr", "section-row");
      const td = el("td", "", `${label} &nbsp;·&nbsp; K = ${curK}`);
      td.colSpan = 13;
      sec.appendChild(td);
      body.appendChild(sec);
      block[c].forEach((row) => body.appendChild(dataRow(row)));
    });

    const w = el("tr", "wins");
    w.appendChild(el("td", "name", tr("Wins of 30")));
    WINS.forEach((n, i) => {
      const cls = [i === 0 ? "ours" : "", GROUP_EDGES.has(i) ? "sep" : ""];
      w.appendChild(el("td", cls.filter(Boolean).join(" "), String(n)));
    });
    w.appendChild(el("td", "ref sep", "—"));
    w.appendChild(el("td", "ref", "—"));
    body.appendChild(w);
  }

  bindChoices("k-seg", "k", (v) => { curK = Number(v); renderMain(); });
  bindChoices("corpus-seg", "c", (v) => { curCorpus = v; renderMain(); });

  /* ------------------------------------------------------- ablation table */
  const ablTable = $("#ablation-table");
  const OOD_EDGE = 7;

  function renderAblation() {
    ablTable.tHead.innerHTML = "";
    const body = ablTable.tBodies[0] || ablTable.createTBody();
    body.innerHTML = "";

    const g = el("tr", "grp");
    g.appendChild(el("th", "name", ""));
    const gi = el("th", "", tr("In-domain corpus")); gi.colSpan = 7; g.appendChild(gi);
    const go = el("th", "sep", tr("Out-of-domain"));  go.colSpan = 3; g.appendChild(go);
    g.appendChild(el("th", "sep", ""));
    ablTable.tHead.appendChild(g);

    const h = el("tr");
    h.appendChild(el("th", "name", tr("Variant")));
    ABLATION_DATASETS.forEach((d, i) => h.appendChild(el("th", i === OOD_EDGE ? "sep" : "", d)));
    h.appendChild(el("th", "sep", tr("Overall")));
    ablTable.tHead.appendChild(h);

    /* Ranked down each column across every variant, as the paper does. */
    const cols = ABLATION_DATASETS.map((_, c) => rankMarks(ABLATION.map((r) => r.v[c])));
    const overall = rankMarks(ABLATION.map((r) => r.overall));

    let group = null;
    ABLATION.forEach((r, ri) => {
      if (r.group && r.group !== group) {
        group = r.group;
        const sec = el("tr", "section-row");
        const td = el("td", "", tr(r.group));
        td.colSpan = 12;
        sec.appendChild(td);
        body.appendChild(sec);
      }

      const row = el("tr");
      const label = tr(r.label);
      row.appendChild(el("td", "name", r.ours ? `<b>${label}</b>` : label));
      r.v.forEach((v, c) => {
        const cls = [cols[c][ri], c === OOD_EDGE ? "sep" : "", r.ours ? "ours" : ""];
        row.appendChild(el("td", cls.filter(Boolean).join(" "), fmt(v)));
      });
      row.appendChild(el("td", [overall[ri], "sep", r.ours ? "ours" : ""].filter(Boolean).join(" "), fmt(r.overall)));
      body.appendChild(row);
    });
  }

  /* --------------------------------------------------------- LT-Lib table */
  const ltTable = $("#ltlib-table");
  const FAMILIES = ["All", ...new Set(LTLIB.map((m) => m.family))];
  let family = "All", sortKey = null, sortDir = "asc";

  const maxima = {
    params: Math.max(...LTLIB.map((m) => m.params)),
    macs:   Math.max(...LTLIB.map((m) => m.macs)),
    lat:    Math.max(...LTLIB.map((m) => m.lat)),
    vram:   Math.max(...LTLIB.map((m) => m.vram)),
  };

  /* Parameter counts span 137 to 2.5M, so that column needs a log scale to
     show anything at all; the rest are linear. A tinted block behind the
     number reads more quietly than a rule beside it. */
  function fillPct(v, key) {
    const frac = key === "params"
      ? Math.log10(v + 1) / Math.log10(maxima.params + 1)
      : v / maxima[key];
    return (3 + frac * 97).toFixed(1);
  }
  const measured = (v, label, key) =>
    `<span class="v">${label}</span>` +
    `<span class="track"><i style="width:${fillPct(v, key)}%"></i></span>`;

  (function buildFamilyFilter() {
    const bar = $("#family-filter");
    FAMILIES.forEach((f) => {
      const b = el("button", "", f);
      b.dataset.family = f;
      if (f === "All") b.setAttribute("aria-pressed", "true");
      bar.appendChild(b);
    });
    bindChoices("family-filter", "family", (v) => { family = v; renderLtlib(); });
  })();

  $$("#ltlib-table th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      sortDir = sortKey === key && sortDir === "asc" ? "desc" : "asc";
      sortKey = key;
      $$("#ltlib-table th.sortable").forEach((x) => x.removeAttribute("data-dir"));
      th.setAttribute("data-dir", sortDir);
      renderLtlib();
    });
  });

  function renderLtlib() {
    $$("#family-filter button").forEach((b) => { b.textContent = tr(b.dataset.family); });

    let rows = LTLIB.filter((m) => family === "All" || m.family === family);
    if (sortKey) {
      rows = [...rows].sort((a, b) =>
        sortDir === "asc" ? a[sortKey] - b[sortKey] : b[sortKey] - a[sortKey]);
    }

    const body = ltTable.tBodies[0] || ltTable.createTBody();
    body.innerHTML = "";
    rows.forEach((m) => {
      const row = el("tr");
      row.appendChild(el("td", "name", m.name));
      row.appendChild(el("td", "left", m.family));
      row.appendChild(el("td", "metric", measured(m.params, m.plabel, "params")));
      row.appendChild(el("td", "metric", measured(m.macs, m.macs.toFixed(2), "macs")));
      row.appendChild(el("td", "metric", measured(m.lat, m.lat.toFixed(2), "lat")));
      row.appendChild(el("td", "metric", measured(m.vram, m.vram.toFixed(1), "vram")));
      row.appendChild(el("td", "left ref",
        m.url ? `<a href="${m.url}" rel="noopener">${m.ref}</a>` : m.ref));
      body.appendChild(row);
    });
  }

  /* --------------------------------------------------------------- start */
  if (lang === "zh") applyLang("zh");
  else { renderMain(); renderAblation(); renderLtlib(); }
})();
