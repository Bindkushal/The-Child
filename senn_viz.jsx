import { useState, useEffect, useRef, useCallback } from "react";

// ── Simulation engine ──────────────────────────────────────────────────────
const INITIAL_LAYERS = [20, 16, 4, 5]; // input, hidden…, output
const MAX_NEURONS = 32;
const MIN_NEURONS = 3;
const NES_THRESHOLD = 0.18;
const PRUNE_THRESHOLD = 0.008;

function initNet(layerSizes) {
  return layerSizes.map((size, li) => ({
    id: li,
    neurons: Array.from({ length: size }, (_, ni) => ({
      id: `${li}-${ni}`,
      activation: Math.random(),
      weight: Math.random() * 2 - 1,
      state: "normal", // normal | growing | pruning | active
      age: Math.floor(Math.random() * 100),
    })),
  }));
}

function stepSimulation(layers, step, task) {
  // Compute fake loss that slowly descends with noise
  const baseLoss = Math.max(0.4, 1.8 - step * 0.0008 + Math.sin(step * 0.1) * 0.15);
  const loss = baseLoss + (Math.random() - 0.5) * 0.12;
  const acc = Math.min(95, 20 + step * 0.03 + Math.random() * 8);

  // NES score per layer (gradient pressure)
  const nes = layers.map(l =>
    l.neurons.reduce((s, n) => s + Math.abs(n.weight), 0) /
    (l.neurons.length * 2 + 1e-6)
  );

  let newLayers = layers.map(l => ({
    ...l,
    neurons: l.neurons.map(n => ({
      ...n,
      activation: Math.max(0, Math.min(1, n.activation + (Math.random() - 0.5) * 0.3)),
      weight: n.weight + (Math.random() - 0.5) * 0.02,
      age: n.state === "growing" ? 1 : n.age + 1,
      state: n.state === "growing" ? (n.age > 8 ? "normal" : "growing")
           : n.state === "pruning" ? (n.age > 6 ? "dead" : "pruning")
           : "normal",
    })).filter(n => n.state !== "dead"),
  }));

  let events = [];

  // Only consider hidden layers (not input/output)
  for (let li = 1; li < newLayers.length - 1; li++) {
    const layer = newLayers[li];
    const nesScore = nes[li];

    // GROW: NES too high and layer not at max
    if (nesScore > NES_THRESHOLD && layer.neurons.length < MAX_NEURONS && Math.random() < 0.35) {
      const addCount = Math.random() < 0.5 ? 1 : 2;
      const newNeurons = Array.from({ length: addCount }, (_, i) => ({
        id: `${li}-new-${Date.now()}-${i}`,
        activation: 0.5,
        weight: (Math.random() - 0.5) * 0.01,
        state: "growing",
        age: 0,
      }));
      newLayers[li] = { ...layer, neurons: [...layer.neurons, ...newNeurons] };
      events.push({ type: "grow", layer: li, count: addCount });
    }

    // PRUNE: weak neurons
    const tooPoor = layer.neurons.filter(n => Math.abs(n.weight) < PRUNE_THRESHOLD && n.age > 20);
    if (tooPoor.length > 0 && layer.neurons.length > MIN_NEURONS) {
      const pruneId = tooPoor[0].id;
      newLayers[li] = {
        ...layer,
        neurons: layer.neurons.map(n =>
          n.id === pruneId ? { ...n, state: "pruning", age: 0 } : n
        ),
      };
      events.push({ type: "prune", layer: li });
    }
  }

  const totalParams = newLayers.reduce((s, l, li) => {
    if (li === 0) return s;
    return s + newLayers[li - 1].neurons.length * l.neurons.length + l.neurons.length;
  }, 0);

  return { layers: newLayers, loss, acc, events, totalParams };
}

// ── Colour helpers ─────────────────────────────────────────────────────────
const STATE_COLOR = {
  normal:  "#38bdf8",
  growing: "#4ade80",
  pruning: "#f87171",
  active:  "#fbbf24",
};

const STATE_GLOW = {
  normal:  "rgba(56,189,248,0.35)",
  growing: "rgba(74,222,128,0.6)",
  pruning: "rgba(248,113,113,0.6)",
  active:  "rgba(251,191,36,0.5)",
};

// ── Main component ─────────────────────────────────────────────────────────
export default function SENNVisualizer() {
  const [layers, setLayers] = useState(() => initNet(INITIAL_LAYERS));
  const [step, setStep] = useState(0);
  const [task, setTask] = useState(1);
  const [metrics, setMetrics] = useState({ loss: 1.8, acc: 20, params: 525 });
  const [log, setLog] = useState([]);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(180);
  const [signals, setSignals] = useState([]);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(null);
  const stateRef = useRef({ layers, step, task });
  stateRef.current = { layers, step, task };

  // Forward-pass signal animation
  const fireSignals = useCallback((layers) => {
    const newSigs = [];
    for (let li = 0; li < layers.length - 1; li++) {
      const src = layers[li].neurons[Math.floor(Math.random() * layers[li].neurons.length)];
      const dst = layers[li + 1].neurons[Math.floor(Math.random() * layers[li + 1].neurons.length)];
      if (src && dst) newSigs.push({ id: Date.now() + li, fromLayer: li, fromId: src.id, toLayer: li + 1, toId: dst.id, progress: 0 });
    }
    setSignals(prev => [...prev.slice(-8), ...newSigs]);
  }, []);

  useEffect(() => {
    if (!running) return;
    let last = performance.now();

    const tick = (now) => {
      if (now - last < speed) { rafRef.current = requestAnimationFrame(tick); return; }
      last = now;

      const { layers: curLayers, step: curStep, task: curTask } = stateRef.current;
      const nextStep = curStep + 1;
      const nextTask = nextStep % 300 === 0 ? Math.min(curTask + 1, 3) : curTask;
      const { layers: nextLayers, loss, acc, events, totalParams } = stepSimulation(curLayers, nextStep, nextTask);

      setStep(nextStep);
      setTask(nextTask);
      setLayers(nextLayers);
      setMetrics({ loss: +loss.toFixed(4), acc: +acc.toFixed(1), params: totalParams });

      if (events.length) {
        setLog(prev => [
          ...prev.slice(-19),
          ...events.map(e => ({
            id: Date.now() + Math.random(),
            step: nextStep,
            text: e.type === "grow"
              ? `↑ Layer ${e.layer}: +${e.count} neuron${e.count > 1 ? "s" : ""}`
              : `↓ Layer ${e.layer}: pruned 1 neuron`,
            type: e.type,
          })),
        ]);
      }

      if (nextStep % 3 === 0) fireSignals(nextLayers);
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, speed, fireSignals]);

  return (
    <div style={{
      minHeight: "100vh", background: "#050a14",
      fontFamily: "'Courier New', monospace",
      color: "#e2e8f0", display: "flex", flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "14px 24px", borderBottom: "1px solid #1e3a5f",
        background: "rgba(5,20,40,0.95)", display: "flex",
        alignItems: "center", gap: 16, backdropFilter: "blur(8px)",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: 2, color: "#38bdf8" }}>
          ◈ SENN
        </span>
        <span style={{ fontSize: 11, color: "#64748b", letterSpacing: 3 }}>
          SELF-EXPANDING NEURAL NETWORK
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
          <MetricBadge label="STEP" value={step} color="#38bdf8" />
          <MetricBadge label="TASK" value={task} color="#818cf8" />
          <MetricBadge label="LOSS" value={metrics.loss} color="#f87171" />
          <MetricBadge label="ACC" value={`${metrics.acc}%`} color="#4ade80" />
          <MetricBadge label="PARAMS" value={metrics.params} color="#fbbf24" />
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Canvas */}
        <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          <NetworkCanvas layers={layers} signals={signals} />
          {/* Legend */}
          <div style={{
            position: "absolute", bottom: 16, left: 16,
            display: "flex", gap: 12, background: "rgba(5,15,30,0.85)",
            border: "1px solid #1e3a5f", borderRadius: 8, padding: "8px 14px",
            backdropFilter: "blur(6px)",
          }}>
            {Object.entries({ ACTIVE: "normal", GROWING: "growing", PRUNING: "pruning" }).map(([label, state]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: STATE_COLOR[state] }} />
                <span style={{ fontSize: 10, color: "#94a3b8", letterSpacing: 1 }}>{label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div style={{
          width: 230, borderLeft: "1px solid #1e3a5f", padding: 16,
          display: "flex", flexDirection: "column", gap: 16,
          background: "rgba(5,12,25,0.95)", overflowY: "auto",
        }}>
          {/* Controls */}
          <Section title="CONTROLS">
            <button
              onClick={() => setRunning(r => !r)}
              style={{
                width: "100%", padding: "8px 0", borderRadius: 6, border: "none",
                background: running ? "#1e3a5f" : "#14532d",
                color: running ? "#38bdf8" : "#4ade80",
                cursor: "pointer", fontFamily: "inherit",
                fontSize: 12, letterSpacing: 2, fontWeight: 700,
              }}
            >
              {running ? "⏸ PAUSE" : "▶ RESUME"}
            </button>
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4, letterSpacing: 1 }}>SPEED</div>
              <input type="range" min={40} max={600} value={600 - speed + 40}
                onChange={e => setSpeed(600 - Number(e.target.value) + 40)}
                style={{ width: "100%", accentColor: "#38bdf8" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#475569" }}>
                <span>SLOW</span><span>FAST</span>
              </div>
            </div>
          </Section>

          {/* Architecture */}
          <Section title="ARCHITECTURE">
            {layers.map((l, li) => (
              <div key={l.id} style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "center", marginBottom: 6,
              }}>
                <span style={{ fontSize: 10, color: "#64748b" }}>
                  {li === 0 ? "INPUT" : li === layers.length - 1 ? "OUTPUT" : `HIDDEN ${li}`}
                </span>
                <div style={{ display: "flex", gap: 3 }}>
                  {Array.from({ length: Math.min(l.neurons.length, 16) }).map((_, ni) => {
                    const n = l.neurons[ni];
                    return (
                      <div key={ni} style={{
                        width: 6, height: 6, borderRadius: "50%",
                        background: n ? STATE_COLOR[n.state] : "#1e293b",
                        opacity: n?.state === "pruning" ? 0.4 : 1,
                        transition: "all 0.3s",
                      }} />
                    );
                  })}
                  {l.neurons.length > 16 && (
                    <span style={{ fontSize: 9, color: "#475569" }}>+{l.neurons.length - 16}</span>
                  )}
                </div>
                <span style={{ fontSize: 11, color: "#38bdf8", fontWeight: 700 }}>
                  {l.neurons.length}
                </span>
              </div>
            ))}
          </Section>

          {/* Event Log */}
          <Section title="EVENT LOG" flex>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 240, overflowY: "auto" }}>
              {[...log].reverse().map(entry => (
                <div key={entry.id} style={{
                  fontSize: 10, padding: "3px 6px", borderRadius: 4,
                  background: entry.type === "grow" ? "rgba(74,222,128,0.08)" : "rgba(248,113,113,0.08)",
                  borderLeft: `2px solid ${entry.type === "grow" ? "#4ade80" : "#f87171"}`,
                  color: entry.type === "grow" ? "#86efac" : "#fca5a5",
                  letterSpacing: 0.5,
                }}>
                  <span style={{ color: "#475569", marginRight: 4 }}>[{entry.step}]</span>
                  {entry.text}
                </div>
              ))}
              {log.length === 0 && (
                <div style={{ fontSize: 10, color: "#334155", fontStyle: "italic" }}>Waiting...</div>
              )}
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}

// ── Network Canvas ─────────────────────────────────────────────────────────
function NetworkCanvas({ layers, signals }) {
  const svgRef = useRef(null);
  const [dims, setDims] = useState({ w: 800, h: 500 });

  useEffect(() => {
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDims({ w: width, h: height });
    });
    if (svgRef.current) obs.observe(svgRef.current.parentElement);
    return () => obs.disconnect();
  }, []);

  const { w, h } = dims;
  const PAD = 60;
  const layerX = layers.map((_, li) => PAD + (li / (layers.length - 1)) * (w - PAD * 2));

  // Compute neuron positions
  const positions = layers.map((layer, li) => {
    const count = layer.neurons.length;
    const spacing = Math.min(36, (h - PAD * 2) / Math.max(count, 1));
    const totalH = spacing * (count - 1);
    const startY = h / 2 - totalH / 2;
    return layer.neurons.map((n, ni) => ({
      ...n, x: layerX[li], y: startY + ni * spacing, li,
    }));
  });

  // Flat lookup for signal rendering
  const lookup = {};
  positions.forEach(lp => lp.forEach(n => { lookup[n.id] = n; }));

  return (
    <svg ref={svgRef} width="100%" height="100%" style={{ display: "block" }}>
      <defs>
        {Object.entries(STATE_COLOR).map(([state, color]) => (
          <radialGradient key={state} id={`g-${state}`} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={color} stopOpacity="1" />
            <stop offset="100%" stopColor={color} stopOpacity="0.2" />
          </radialGradient>
        ))}
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="bigGlow">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* Background grid */}
      {Array.from({ length: 20 }).map((_, i) => (
        <line key={`h${i}`} x1={0} y1={(i / 19) * h} x2={w} y2={(i / 19) * h}
          stroke="#0f172a" strokeWidth={1} />
      ))}
      {Array.from({ length: 30 }).map((_, i) => (
        <line key={`v${i}`} x1={(i / 29) * w} y1={0} x2={(i / 29) * w} y2={h}
          stroke="#0f172a" strokeWidth={1} />
      ))}

      {/* Connections */}
      {positions.slice(0, -1).map((srcLayer, li) =>
        srcLayer.slice(0, 20).map(src =>
          positions[li + 1].slice(0, 20).map(dst => (
            <line
              key={`${src.id}-${dst.id}`}
              x1={src.x} y1={src.y} x2={dst.x} y2={dst.y}
              stroke="#1e3a5f" strokeWidth={0.5} opacity={0.4}
            />
          ))
        )
      )}

      {/* Signal pulses */}
      {signals.map(sig => {
        const from = lookup[sig.fromId];
        const to = lookup[sig.toId];
        if (!from || !to) return null;
        return (
          <SignalDot key={sig.id} x1={from.x} y1={from.y} x2={to.x} y2={to.y} />
        );
      })}

      {/* Neurons */}
      {positions.map(lp => lp.map(n => {
        const r = n.state === "growing" ? 10 : n.state === "pruning" ? 5 : 8;
        const color = STATE_COLOR[n.state] || "#38bdf8";
        return (
          <g key={n.id} filter={n.state !== "normal" ? "url(#bigGlow)" : "url(#glow)"}>
            {/* Outer ring for growing/pruning */}
            {n.state !== "normal" && (
              <circle cx={n.x} cy={n.y} r={r + 5}
                fill="none" stroke={color} strokeWidth={1.5} opacity={0.4}
                style={{ animation: "pulse 1s ease-in-out infinite" }}
              />
            )}
            <circle cx={n.x} cy={n.y} r={r}
              fill={`url(#g-${n.state})`}
              stroke={color} strokeWidth={1.2}
              opacity={n.state === "pruning" ? 0.5 : 1}
            />
            {/* Activation fill */}
            <circle cx={n.x} cy={n.y} r={r * n.activation * 0.7}
              fill={color} opacity={0.6}
            />
          </g>
        );
      }))}

      {/* Layer labels */}
      {layers.map((l, li) => (
        <text key={li} x={layerX[li]} y={PAD - 16}
          textAnchor="middle" fontSize={9} fill="#334155" letterSpacing={1}
          fontFamily="'Courier New', monospace"
        >
          {li === 0 ? "IN" : li === layers.length - 1 ? "OUT" : `H${li}`}
          ({l.neurons.length})
        </text>
      ))}

      <style>{`
        @keyframes pulse { 0%,100% { opacity:0.2; r:13; } 50% { opacity:0.7; r:16; } }
      `}</style>
    </svg>
  );
}

// ── Animated signal dot ────────────────────────────────────────────────────
function SignalDot({ x1, y1, x2, y2 }) {
  const [t, setT] = useState(0);
  useEffect(() => {
    let raf;
    let start = performance.now();
    const dur = 500;
    const step = (now) => {
      const p = Math.min(1, (now - start) / dur);
      setT(p);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);
  const x = x1 + (x2 - x1) * t;
  const y = y1 + (y2 - y1) * t;
  return (
    <circle cx={x} cy={y} r={3} fill="#fbbf24" opacity={1 - t * 0.6}
      filter="url(#glow)" />
  );
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function MetricBadge({ label, value, color }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      background: "rgba(14,30,55,0.8)", border: `1px solid ${color}22`,
      borderRadius: 6, padding: "4px 10px", minWidth: 60,
    }}>
      <span style={{ fontSize: 8, color: "#475569", letterSpacing: 1 }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </span>
    </div>
  );
}

function Section({ title, children, flex }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", flex: flex ? 1 : undefined }}>
      <div style={{
        fontSize: 9, letterSpacing: 2, color: "#334155",
        borderBottom: "1px solid #1e3a5f", paddingBottom: 4, marginBottom: 8,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}
