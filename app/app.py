"""
CAI Application - Radiology worklist demo (Streamlit).

  - shows films in data/incoming/ in arrival order (FIFO), then ranks them (and
    any uploads): P1, P2 by the deciding champion's probability, then P3 and NA (no AI
    triage: the film check rejected the film, or no champion covers the patient) in
    arrival order
  - shows an occlusion heatmap for the selected film, for the model that ranked it
  - captures the radiologist's agree/override as feedback for the next retrain
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_config  # noqa: E402
from features.feature_logic import load_image, preprocess  # noqa: E402

st.set_page_config(page_title="CXR Triage - Cloudera AI", layout="wide")
cfg = load_config()
BAND_COLOR = {"P1": "#FF550C", "P2": "#FE8756", "P3": "#A8AFB9", "NA": "#5B6B7F"}
BAND_TEXT = {"P1": "read first", "P2": "likely abnormal", "P3": "routine", "NA": "no AI triage"}
FLAGGED = ("P1", "P2")
SUFFIXES = {".jpeg", ".jpg", ".png"}
MAX_FILMS = 40


@st.cache_resource
def engine():
    import serve.predict as p
    return p


try:
    eng = engine()
except RuntimeError as e:
    st.title("Chest X-ray triage worklist")
    st.error(f"No model to serve yet: {e}")
    st.stop()
meta = eng._META


def deciding(r: dict) -> tuple[str, float]:
    """The model that set the band and its probability (pneumonia when none did)."""
    name = r.get("triage_model") or "pneumonia"
    return name, r["findings"][name]["probability"] if name in r.get("findings", {}) else r["probability_pneumonia"]


def worklist_order(rows: list[dict]) -> list[dict]:
    """P1 then P2, higher probability first; then P3 and NA in arrival order."""
    def key(ir):
        i, r = ir
        flagged = r["priority"] in FLAGGED
        return (FLAGGED.index(r["priority"]) if flagged else len(FLAGGED), -deciding(r)[1] if flagged else i)
    return [r for _, r in sorted(enumerate(rows), key=key)]


st.title("Chest X-ray triage worklist")
st.caption(f"Decision support only - every film is still read by a radiologist.  "
           f"Model git {meta['git_sha'][:7]} - features v{meta['feature_version']} - "
           f"test AUROC {meta['metrics']['test']['auroc']:.3f} - "
           f"sensitivity {meta['metrics']['test']['sensitivity']:.3f} at threshold {meta['threshold']:.3f}.  "
           f"Champions: {', '.join(sorted(eng._CHAMPIONS))}")

incoming_dir = ROOT / cfg["data"]["incoming_dir"]
incoming = sorted(p for p in incoming_dir.rglob("*") if p.suffix.lower() in SUFFIXES)[:MAX_FILMS]

left, right = st.columns([2, 3])
with left:
    uploads = st.file_uploader("Add films", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Triage worklist", type="primary"):
        items = [(p.name, load_image(p)) for p in incoming] + [(u.name, load_image(u.getvalue())) for u in uploads or []]
        scores = eng.score_images([im for _, im in items]) if items else []
        st.session_state.worklist = worklist_order(
            [{"film": n, "image": im, **s} for (n, im), s in zip(items, scores)])
        st.session_state.sel = 0

    wl = st.session_state.get("worklist", [])
    if not wl:
        st.subheader(f"Arrival order (FIFO) - {len(incoming)} films")
        if not incoming:
            st.info(f"No films in {cfg['data']['incoming_dir']}/ - copy some test films there or upload.")
        for i, p in enumerate(incoming, 1):
            st.text(f"{i:>2}. {p.name}")
    else:
        counts = {b: sum(r["priority"] == b for r in wl) for b in BAND_COLOR}
        cols = st.columns(len(BAND_COLOR))
        for c, (b, n) in zip(cols, counts.items()):
            c.metric(f"{b} ({BAND_TEXT[b]})", n)
        for i, r in enumerate(wl):
            name, prob = deciding(r)
            label = f"{r['priority']}  {name} {prob:.2f}  {r['film']}" if r["triage_model"] else \
                f"{r['priority']}  {r['film']}"
            if (r.get("film_qc") or {}).get("unsuitable"):
                label += "  (film unfit for AI triage)"
            elif r["quality_flags"]:
                label += "  (check quality)"
            if st.button(label, key=f"row{i}"):
                st.session_state.sel = i

with right:
    wl = st.session_state.get("worklist", [])
    if wl:
        r = wl[min(st.session_state.get("sel", 0), len(wl) - 1)]
        st.markdown(f"### {r['film']}  <span style='color:{BAND_COLOR[r['priority']]}'>{r['priority']}</span>",
                    unsafe_allow_html=True)
        if st.checkbox("Show occlusion heatmap (about 10 s on CPU)", key=f"heat-{r['film']}"):
            import matplotlib.cm as cm

            from serve.explain import occlusion_map

            model = deciding(r)[0]

            def score_fn(imgs):
                return np.array([s["findings"][model]["probability"] for s in eng.score_images(imgs)])

            with st.spinner("Occluding patches..."):
                heat, _ = occlusion_map(r["image"], score_fn, cfg["features"]["image_size"])
            base_img = np.asarray(preprocess(r["image"], cfg["features"]["image_size"]).convert("RGB")) / 255.0
            overlay = 0.6 * base_img + 0.4 * cm.inferno(heat)[..., :3]
            st.image(Image.fromarray((overlay * 255).astype(np.uint8)), width=448,
                     caption="Indicative only: regions whose occlusion lowers the score most")
        else:
            st.image(r["image"], width=448)
        st.json({k: r[k] for k in ("priority", "triage_model", "film_qc", "findings", "quality_flags")})
        verdict = st.radio("Radiologist read", ["Agree", "Override: NORMAL", "Override: PNEUMONIA",
                                                "Override: PNEUMOTHORAX"],
                           horizontal=True, key=f"read-{r['film']}")
        if st.button("Save read"):
            fb = ROOT / "outputs" / "feedback" / "feedback.csv"
            fb.parent.mkdir(parents=True, exist_ok=True)
            new = not fb.exists()
            with open(fb, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["ts_utc", "film", "model_prob", "model_priority", "radiologist_read",
                                "model_git_sha", "feature_version", "triage_model"])
                w.writerow([datetime.now(timezone.utc).isoformat(), r["film"], deciding(r)[1],
                            r["priority"], verdict, meta["git_sha"][:7], meta["feature_version"], r["triage_model"]])
            st.success("Saved - feeds the next labelled batch")
