import { useState } from "react";

const LEGEND = [
  ["#00f", "Impacted"], ["#0ff", "Caries"],
  ["#ff0", "Periapical"], ["#0f0", "Deep Caries"],
];

export default function App() {
  const [status, setStatus] = useState("");
  const [analysisId, setAnalysisId] = useState(null);
  const [overview, setOverview] = useState(null);
  const [findings, setFindings] = useState([]);
  const [active, setActive] = useState(null);   // null = overview; else seg result
  const [activeIdx, setActiveIdx] = useState(null);

  async function upload(file) {
    if (!file) return;
    setStatus("⏳ Running YOLOv8 + SAM adapter… (first run loads models)");
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const d = await r.json();
    if (d.error) { setStatus("❌ " + d.error); return; }
    setAnalysisId(d.id);
    setOverview(d.overview);
    setFindings(d.findings);
    setActive(null); setActiveIdx(null);
    setStatus(`✅ ${d.findings.length} lesion(s) detected.`);
  }

  async function segment(idx) {
    setActiveIdx(idx);
    setActive("loading");
    const r = await fetch("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: analysisId, idx }),
    });
    setActive(await r.json());
  }

  return (
    <>
      <header>
        <h1>🦷 OPG-Live</h1>
        <span className="tag">MVP · YOLOv8 detect → SAM Adapter segment</span>
      </header>

      <main>
        <div className="col">
          <section className="card">
            <h2>1 · Upload OPG</h2>
            <label
              className="drop"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); upload(e.dataTransfer.files[0]); }}
            >
              <input type="file" accept="image/*" hidden
                onChange={(e) => upload(e.target.files[0])} />
              <div>Click or drop a panoramic image</div>
            </label>
            <p className="muted">{status}</p>
          </section>

          <section className="card">
            <h2>Findings
              <button onClick={() => { setActive(null); setActiveIdx(null); }}>▦ Overview (all)</button>
            </h2>
            {findings.length === 0
              ? <p className="muted">Upload an OPG to detect lesions.</p>
              : findings.map((f) => (
                <div key={f.idx}
                  className={"finding" + (activeIdx === f.idx ? " active" : "")}
                  onClick={() => segment(f.idx)}>
                  <div><span className="dx">{f.disease}</span>
                    <span className="meta"> · {f.fdi ? "FDI " + f.fdi : "tooth ?"} · {f.conf}</span></div>
                  <span className="pill">segment →</span>
                </div>
              ))}
          </section>
        </div>

        <section className="card viewer">
          {active === null && (
            <>
              <h2>Overview — all detections</h2>
              {overview
                ? <><img className="result" src={overview} alt="overview" />
                    <div className="legend">{LEGEND.map(([c, n]) =>
                      <span key={n}><i className="sw" style={{ background: c }} />{n}</span>)}</div></>
                : <p className="muted">Result appears here.</p>}
            </>
          )}
          {active === "loading" && <p className="muted">⏳ rendering mask…</p>}
          {active && active !== "loading" && !active.error && (
            <>
              <h2>{active.disease}{active.fdi ? " · FDI " + active.fdi : ""}</h2>
              <div className="grid2">
                <img className="result" src={active.view} alt="segmentation" />
                <div><img className="result" src={active.crop} alt="zoom" />
                  <p className="muted center">zoom</p></div>
              </div>
              <div className="metrics">
                <span>diagnosis<b>{active.disease}</b></span>
                <span>tooth (FDI)<b>{active.fdi || "?"}</b></span>
                <span>detector conf<b>{active.conf}</b></span>
                <span>mask area (px)<b>{active.mask_area_px?.toLocaleString()}</b></span>
                <span>SAM score<b>{active.sam_score}</b></span>
              </div>
            </>
          )}
          {active?.error && <p className="muted">⚠️ {active.error}</p>}
        </section>
      </main>
    </>
  );
}
