"""
app.py
======
Hugging Face Space — The AI Child
Gradio interface with Draw / Upload / Chat / Brain tabs.

Hosted at: huggingface.co/spaces/Bindkushal/the-child
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from PIL import Image, ImageOps, ImageFilter
from datetime import datetime

# ── Try to import the network ─────────────────────────────────────────────
try:
    from dynamic_net import SelfExpandingNet
    NET_AVAILABLE = True
except ImportError:
    NET_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────
LETTERS      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODEL_PATH   = "senn_final.pt"
ARCH_PATH    = "architecture_final.json"
JOURNAL_PATH = "growth_journal.json"

# ── Load model ────────────────────────────────────────────────────────────
net  = None
arch = None

def load_model():
    global net, arch
    if not NET_AVAILABLE:
        return "dynamic_net.py not found"
    if not os.path.exists(MODEL_PATH):
        return "senn_final.pt not found — train first"
    if not os.path.exists(ARCH_PATH):
        return "architecture_final.json not found — train first"
    try:
        with open(ARCH_PATH) as f:
            arch = json.load(f)
        hidden = [l["out"] for l in arch["hidden_layers"]]
        net = SelfExpandingNet(
            input_size    = arch["input_size"],
            output_size   = arch["output_size"],
            initial_hidden= hidden
        )
        state = torch.load(MODEL_PATH, map_location="cpu")
        net.load_state_dict(state)
        net.eval()
        return "ok"
    except Exception as e:
        return str(e)

_status = load_model()

# ── Image preprocessing ───────────────────────────────────────────────────
def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Convert any input image to the 784-dim tensor the model expects.
    EMNIST format: white letter on black background.
    """
    img = img.convert("L")                          # grayscale
    img = ImageOps.invert(img)                      # invert: dark bg → white letter
    img = img.resize((28, 28), Image.LANCZOS)       # resize
    img = img.filter(ImageFilter.SHARPEN)           # sharpen edges
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize
    return torch.tensor(arr).view(1, 784)

def predict(tensor: torch.Tensor):
    """Run inference, return top-5 predictions."""
    if net is None:
        return None
    with torch.no_grad():
        out   = net(tensor)
        probs = F.softmax(out, dim=-1)[0]
        topk  = probs.topk(5)
    results = []
    for prob, idx in zip(topk.values, topk.indices):
        results.append((LETTERS[idx.item()], round(prob.item() * 100, 2)))
    return results

def format_predictions(results):
    """Format prediction results as a nice string."""
    if not results:
        return "⚠ Model not loaded. Please upload senn_final.pt to the Space files."
    lines = ["### Predictions\n"]
    for i, (letter, pct) in enumerate(results):
        bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        medal = ["🥇", "🥈", "🥉", "4.", "5."][i]
        lines.append(f"{medal} **{letter}** — {bar} {pct:.1f}%")
    return "\n".join(lines)

# ── CSS — dark neural theme matching brain.html ───────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg:      #020915;
    --bg2:     #040f20;
    --border:  #0d2038;
    --cyan:    #00e5ff;
    --green:   #39ff14;
    --amber:   #ffcc00;
    --purple:  #bb55ff;
    --text:    #b8ccd8;
    --dim:     #4a6070;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header */
.senn-header {
    text-align: center;
    padding: 28px 0 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.senn-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 8px;
    text-shadow: 0 0 30px rgba(0,229,255,0.4);
    margin: 0;
}
.senn-sub {
    font-size: 0.7rem;
    color: var(--dim);
    letter-spacing: 4px;
    margin-top: 4px;
}
.senn-status {
    font-size: 0.75rem;
    margin-top: 10px;
    color: var(--green);
}

/* Tabs */
.tab-nav button {
    background: var(--bg2) !important;
    color: var(--dim) !important;
    border: 1px solid var(--border) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
    font-size: 0.75rem !important;
}
.tab-nav button.selected {
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
    background: rgba(0,229,255,0.05) !important;
}

/* Inputs / outputs */
.gr-box, .gr-form, .gr-panel, textarea, .gr-input {
    background: var(--bg2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #003344, #006688) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #006688, #00a8cc) !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.3) !important;
}

/* Markdown output */
.gr-markdown {
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 12px 0;
}
.stat-card {
    background: rgba(0,18,40,0.8);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    text-align: center;
}
.stat-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--cyan);
}
.stat-label {
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 2px;
    margin-top: 2px;
}
"""

# ── BRAIN TAB — stats from arch + journal ─────────────────────────────────
def get_brain_report():
    if arch is None:
        return "⚠ Model not loaded. Upload senn_final.pt and architecture_final.json to the Space."

    hidden_sizes = [l["out"] for l in arch["hidden_layers"]]
    arch_str     = f"{arch['input_size']} → {' → '.join(map(str, hidden_sizes))} → {arch['output_size']}"
    total_params = arch["total_params"]
    growth_count = arch["growth_events"]

    # Load growth journal if available
    journal_lines = []
    if os.path.exists(JOURNAL_PATH):
        try:
            with open(JOURNAL_PATH) as f:
                journal = json.load(f)
            journal_lines.append(f"\n### Growth Journal — last {min(10, len(journal))} events\n")
            for entry in journal[-10:]:
                ts  = entry.get("timestamp", "")[:19].replace("T", " ")
                evt = entry.get("event", {})
                why = entry.get("meaning", "")
                journal_lines.append(f"**Gen {entry['generation']}** `{ts}`")
                journal_lines.append(f"  ↑ {evt.get('type','growth')} — layer {evt.get('layer','?')}, +{evt.get('added','?')} neurons")
                journal_lines.append(f"  _{why}_\n")
        except Exception:
            pass

    report = f"""
## ⬡ SENN Architecture Report

---

### Current Network

```
{arch_str}
```

| Metric | Value |
|--------|-------|
| Total Parameters | {total_params:,} |
| Hidden Layers | {len(hidden_sizes)} |
| Growth Events | {growth_count} |
| Input Size | {arch['input_size']} (28×28 pixels) |
| Output Classes | {arch['output_size']} (A–Z) |

### Layer Breakdown

| Layer | Type | Neurons |
|-------|------|---------|
| 0 | Input | {arch['input_size']} |
"""
    for i, l in enumerate(arch["hidden_layers"]):
        report += f"| {i+1} | Hidden | {l['out']} |\n"
    report += f"| {len(arch['hidden_layers'])+1} | Output | {arch['output_size']} |\n"

    if journal_lines:
        report += "\n" + "\n".join(journal_lines)

    return report


# ── CHAT TAB ───────────────────────────────────────────────────────────────
CHAT_SYSTEM = """You are the SENN AI Child — a self-expanding neural network that has learned 
to recognise handwritten letters A-Z. You started with only 16 neurons and grew your own architecture 
through training. Speak in first person, be curious and child-like but technically precise.
Answer questions about your architecture, how you learn, your growth history, EWC memory, 
and what you find difficult. Keep answers concise and interesting."""

def chat_respond(message, history):
    if not message.strip():
        return history, ""

    # Build context from arch
    ctx = ""
    if arch:
        hidden = [l["out"] for l in arch["hidden_layers"]]
        ctx = (f"My current architecture: {arch['input_size']} → "
               f"{' → '.join(map(str,hidden))} → {arch['output_size']}. "
               f"I have {arch['total_params']:,} parameters and grew {arch['growth_events']} times.")

    # Build messages for Claude API
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user",    "content": user_msg})
        messages.append({"role": "assistant","content": bot_msg})
    messages.append({"role": "user", "content": message})

    system = CHAT_SYSTEM + (f"\n\nCurrent state: {ctx}" if ctx else "")

    try:
        import anthropic
        client   = anthropic.Anthropic()
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 300,
            system     = system,
            messages   = messages
        )
        reply = response.content[0].text
    except Exception:
        # Fallback: rule-based responses when API not available
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["how many", "parameter", "size", "big"]):
            reply = ctx if ctx else "I haven't been trained yet — no weights loaded."
        elif any(w in msg_lower for w in ["grow", "expand", "neuron"]):
            if arch:
                reply = (f"I grew {arch['growth_events']} times during training! "
                         f"I started with just 16 neurons and expanded when I was struggling. "
                         f"The Natural Expansion Score told me when to add more capacity.")
            else:
                reply = "I haven't grown yet — train me first!"
        elif any(w in msg_lower for w in ["forget", "memory", "ewc"]):
            reply = ("I use Elastic Weight Consolidation to remember old tasks. "
                     "It puts a spring on important weights so they can't drift too far "
                     "from what made me good at previous tasks.")
        elif any(w in msg_lower for w in ["difficult", "hard", "struggle"]):
            reply = ("I find similar-looking letters hardest — like I, J, and L. "
                     "Also Q and O. Their pixel patterns overlap a lot.")
        elif any(w in msg_lower for w in ["hello", "hi", "hey"]):
            reply = "Hello! I'm SENN — a self-expanding neural network. I learned to read handwritten letters by growing my own brain. Ask me anything about how I work!"
        else:
            reply = ("Interesting question! I'm SENN, a neural network that grows itself. "
                     "I can tell you about my architecture, memory system, or how I recognise letters. What would you like to know?")

    history.append((message, reply))
    return history, ""


# ── BUILD UI ──────────────────────────────────────────────────────────────
model_loaded = net is not None
status_text  = f"● MODEL LOADED — {arch['total_params']:,} params" if model_loaded else "○ NO MODEL — upload senn_final.pt to files"
status_color = "#39ff14" if model_loaded else "#ff2255"

with gr.Blocks(css=CSS, title="SENN — The AI Child") as demo:

    # Header
    gr.HTML(f"""
    <div class="senn-header">
        <h1 class="senn-title">⬡ SENN</h1>
        <div class="senn-sub">SELF-EXPANDING NEURAL NETWORK — THE AI CHILD</div>
        <div class="senn-status" style="color:{status_color}">{status_text}</div>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: DRAW ──────────────────────────────────────────────────
        with gr.Tab("✏ DRAW"):
            gr.Markdown("### Draw a letter — use your mouse or touchscreen")
            with gr.Row():
                with gr.Column(scale=1):
                    sketchpad = gr.Sketchpad(
                        label      = "Draw here",
                        brush      = gr.Brush(default_size=18, colors=["#ffffff"]),
                        canvas_size= 280,
                        image_mode = "L",
                    )
                    draw_btn = gr.Button("⬡ RECOGNISE", variant="primary")
                    gr.Markdown("*Draw one capital letter, thick strokes work best*")

                with gr.Column(scale=1):
                    draw_output = gr.Markdown("*Draw a letter and click Recognise*")

            def recognise_drawing(sketch):
                if sketch is None:
                    return "⚠ Draw something first"
                try:
                    img     = Image.fromarray(sketch.astype(np.uint8))
                    tensor  = preprocess(img)
                    results = predict(tensor)
                    return format_predictions(results)
                except Exception as e:
                    return f"Error: {e}"

            draw_btn.click(recognise_drawing, inputs=sketchpad, outputs=draw_output)

        # ── TAB 2: UPLOAD ────────────────────────────────────────────────
        with gr.Tab("📷 UPLOAD"):
            gr.Markdown("### Upload a photo of a handwritten letter")
            with gr.Row():
                with gr.Column(scale=1):
                    upload_img = gr.Image(
                        label  = "Upload image",
                        type   = "pil",
                        height = 280,
                    )
                    upload_btn = gr.Button("⬡ RECOGNISE", variant="primary")
                    gr.Markdown("*Tips: good lighting, dark pen on white paper, letter centered*")

                with gr.Column(scale=1):
                    upload_output  = gr.Markdown("*Upload an image and click Recognise*")
                    upload_preview = gr.Image(label="What the network sees (28×28)", height=140)

            def recognise_upload(img):
                if img is None:
                    return "⚠ Upload an image first", None
                try:
                    tensor  = preprocess(img)
                    results = predict(tensor)
                    # Show the 28x28 preview
                    preview = Image.fromarray(
                        (tensor.view(28, 28).numpy() * 255).astype(np.uint8)
                    ).resize((112, 112), Image.NEAREST)
                    return format_predictions(results), preview
                except Exception as e:
                    return f"Error: {e}", None

            upload_btn.click(recognise_upload,
                             inputs=upload_img,
                             outputs=[upload_output, upload_preview])

        # ── TAB 3: CHAT ──────────────────────────────────────────────────
        with gr.Tab("💬 CHAT"):
            gr.Markdown("### Talk to SENN — ask about its architecture, memory, or how it learns")
            chatbot  = gr.Chatbot(height=380, bubble_full_width=False)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder = "Ask me anything — how big am I? what do I find hard? how does my memory work?",
                    show_label  = False,
                    scale       = 5
                )
                chat_send = gr.Button("SEND", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    ["How many neurons do you have?"],
                    ["How do you grow yourself?"],
                    ["What letters do you find hardest?"],
                    ["How do you avoid forgetting old tasks?"],
                    ["Explain your architecture like I'm 10"],
                ],
                inputs=chat_input
            )

            chat_send.click(chat_respond,
                            inputs=[chat_input, chatbot],
                            outputs=[chatbot, chat_input])
            chat_input.submit(chat_respond,
                              inputs=[chat_input, chatbot],
                              outputs=[chatbot, chat_input])

        # ── TAB 4: BRAIN ─────────────────────────────────────────────────
        with gr.Tab("🧠 BRAIN"):
            gr.Markdown("### Architecture report — how SENN grew during training")
            refresh_btn  = gr.Button("↻ REFRESH REPORT", variant="primary")
            brain_output = gr.Markdown(get_brain_report())
            refresh_btn.click(get_brain_report, outputs=brain_output)


if __name__ == "__main__":
    demo.launch()
