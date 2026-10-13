// Voxwerk voice browser: loads voices.json (built by scripts/build_site.py) and renders a searchable, filterable list.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const els = { q: $("q"), region: $("region"), category: $("category"), style: $("style"), gender: $("gender"), age: $("age") };
  const list = $("voices"), count = $("count"), empty = $("empty"), sentinel = $("sentinel"), tpl = $("card");
  const PAGE = 48;
  const AGE_ORDER = ["young", "middle-aged", "old"];
  let voices = [], matches = [], shown = 0;

  // One shared player: only one sample plays at a time.
  const audio = new Audio();
  audio.preload = "none";
  let playingCard = null;

  function haystack(v) {
    return [v.id, v.name, v.description, v.category, v.region, v.style, v.gender, v.age, v.design_prompt, ...(v.tags || [])]
      .join(" ").toLowerCase();
  }

  function fillSelect(sel, values, label = (x) => x) {
    const keep = sel.value;
    sel.length = 1;
    for (const v of values) sel.add(new Option(label(v), v));
    sel.value = values.includes(keep) ? keep : "";
  }

  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const uniq = (arr) => [...new Set(arr.filter(Boolean))];

  function refreshCategories() {
    const region = els.region.value;
    const cats = uniq(voices.filter((v) => !region || v.region === region).map((v) => v.category)).sort();
    fillSelect(els.category, cats);
  }

  function readState() {
    const p = new URLSearchParams(location.search);
    for (const [k, el] of Object.entries(els)) el.value = p.get(k) || "";
  }

  function writeState() {
    const p = new URLSearchParams();
    for (const [k, el] of Object.entries(els)) if (el.value) p.set(k, el.value);
    const qs = p.toString();
    history.replaceState(null, "", (qs ? "?" + qs : location.pathname) + location.hash);
  }

  function apply() {
    const terms = els.q.value.toLowerCase().split(/\s+/).filter(Boolean);
    const f = { region: els.region.value, category: els.category.value, style: els.style.value, gender: els.gender.value, age: els.age.value };
    matches = voices.filter((v) =>
      (!f.region || v.region === f.region) && (!f.category || v.category === f.category) && (!f.style || v.style === f.style) &&
      (!f.gender || v.gender === f.gender) && (!f.age || v.age === f.age) && terms.every((t) => v._hay.includes(t)));
    list.replaceChildren();
    shown = 0;
    more();
    count.textContent = `${matches.length.toLocaleString()} of ${voices.length.toLocaleString()} voices`;
    empty.hidden = matches.length > 0;
    writeState();
  }

  function more() {
    const frag = document.createDocumentFragment();
    for (const v of matches.slice(shown, shown + PAGE)) frag.append(card(v));
    shown = Math.min(matches.length, shown + PAGE);
    list.append(frag);
  }

  function card(v) {
    const li = tpl.content.firstElementChild.cloneNode(true);
    li.id = v.id;
    li.dataset.id = v.id;
    li.querySelector(".name").textContent = v.name;
    const idBtn = li.querySelector(".id");
    idBtn.textContent = v.id;
    idBtn.addEventListener("click", () => copy(v.id, idBtn));
    const meta = li.querySelector(".meta");
    const parts = [v.category, v.style, v.gender, v.age].filter(Boolean);
    meta.innerHTML = parts.map((p, i) => i === 0 ? `<b>${esc(p)}</b>` : esc(p)).join(" · ") +
      (v.sample_seconds ? ` · ${v.sample_seconds.toFixed(1)} s` : "");
    li.querySelector(".desc").textContent = v.description;
    li.querySelector(".sample").textContent = `“${v.sample_text}”`;
    li.querySelector(".prompt").textContent = v.design_prompt;
    li.querySelector(".pkg").textContent = v.sample_package || "—";
    li.querySelector(".usage").textContent =
      `voxwerk design --voice-id ${v.id} --text "Hello there!"   # from the design prompt\n` +
      `voxwerk synth  --voice-id ${v.id} --text "Hello there!"   # clone the sample (needs its audio package)`;
    const tags = li.querySelector(".tags");
    for (const t of v.tags || []) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = t;
      b.addEventListener("click", () => { els.q.value = t; apply(); window.scrollTo({ top: $("toolbar").offsetTop, behavior: "smooth" }); });
      const item = document.createElement("li");
      item.append(b);
      tags.append(item);
    }
    const play = li.querySelector(".play");
    if (v.sample) {
      play.setAttribute("aria-label", `Play ${v.name}`);
      play.addEventListener("click", () => toggle(v, li));
    } else {
      play.disabled = true;
      play.title = "No sample";
    }
    return li;
  }

  function toggle(v, li) {
    if (playingCard === li) {
      if (audio.paused) audio.play(); else audio.pause();
      return;
    }
    stop();
    playingCard = li;
    li.classList.add("loading");
    audio.src = v.sample;
    audio.play().catch(() => {
      li.classList.remove("loading");
      li.querySelector(".play").title = "This browser cannot play Ogg Opus audio.";
    });
  }

  function stop() {
    if (!playingCard) return;
    audio.pause();
    playingCard.classList.remove("playing", "paused", "loading");
    playingCard.querySelector(".progress span").style.width = "0";
    playingCard = null;
  }

  audio.addEventListener("playing", () => playingCard && (playingCard.classList.remove("loading", "paused"), playingCard.classList.add("playing")));
  audio.addEventListener("pause", () => playingCard && (playingCard.classList.remove("playing"), playingCard.classList.add("paused")));
  audio.addEventListener("ended", stop);
  audio.addEventListener("timeupdate", () => {
    if (playingCard && audio.duration) playingCard.querySelector(".progress span").style.width = `${(100 * audio.currentTime) / audio.duration}%`;
  });

  function copy(text, btn) {
    const done = () => { btn.classList.add("copied"); setTimeout(() => btn.classList.remove("copied"), 1200); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => select(btn));
    else select(btn);
  }

  function select(node) {
    const r = document.createRange();
    r.selectNodeContents(node);
    const s = getSelection();
    s.removeAllRanges();
    s.addRange(r);
  }

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // Deep link: #<voice-id> shows that voice even when it is beyond the first page.
  function reveal() {
    const id = decodeURIComponent(location.hash.slice(1));
    if (!id) return;
    const idx = matches.findIndex((v) => v.id === id);
    if (idx < 0) return;
    while (shown <= idx) more();
    const el = document.getElementById(id);
    if (!el) return;
    document.querySelectorAll(".voice.target").forEach((n) => n.classList.remove("target"));
    el.classList.add("target");
    el.scrollIntoView({ block: "center" });
  }

  let timer = 0;
  els.q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(apply, 120); });
  els.region.addEventListener("change", () => { refreshCategories(); apply(); });
  for (const k of ["category", "style", "gender", "age"]) els[k].addEventListener("change", apply);
  $("reset").addEventListener("click", () => { for (const el of Object.values(els)) el.value = ""; refreshCategories(); apply(); });
  window.addEventListener("hashchange", reveal);
  new IntersectionObserver((e) => { if (e[0].isIntersecting && shown < matches.length) more(); }, { rootMargin: "800px" }).observe(sentinel);

  fetch("voices.json")
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((data) => {
      voices = data.map((v) => ({ ...v, _hay: haystack(v) }));
      fillSelect(els.region, uniq(voices.map((v) => v.region)).sort());
      fillSelect(els.style, uniq(voices.map((v) => v.style)).sort());
      fillSelect(els.gender, uniq(voices.map((v) => v.gender)).sort(), cap);
      fillSelect(els.age, AGE_ORDER.filter((a) => voices.some((v) => v.age === a)), cap);
      readState();
      refreshCategories();
      readState();
      apply();
      reveal();
    })
    .catch((e) => { count.textContent = `Could not load the voice list (${e.message}).`; });
})();
