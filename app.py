"""Interactive Real-Time Chat Moderation App (Gradio) — SWE015 Use Case 4.

Run locally : python app.py
Run in Colab: python app.py --share
"""
import argparse
import html
import inspect

import gradio as gr

from src.moderation import DEFAULT_MODEL, LATENCY_BUDGET_MS, MODEL_CHOICES, ModerationEngine

CUSTOM_CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
.mod-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #172554 100%);
    color: #f8fafc; padding: 18px 22px; border-radius: 14px;
    border: 1px solid #334155; margin-bottom: 10px;
}
.mod-hero h1 { margin: 0 0 6px 0; font-size: 1.45rem; font-weight: 700; color: #ffffff; }
.mod-hero p { margin: 0 0 10px 0; color: #cbd5e1; font-size: 0.92rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; background: rgba(56, 189, 248, 0.16);
    color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.35);
}
.pill-green { background: rgba(34, 197, 94, 0.16); color: #86efac; border-color: rgba(34, 197, 94, 0.35); }
.pill-amber { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border-color: rgba(245, 158, 11, 0.35); }
.mod-card {
    background: #0f172a; color: #e2e8f0; border: 1px solid #334155;
    border-radius: 12px; padding: 14px 16px; font-size: 0.88rem; line-height: 1.5;
}
.mod-card code { background: #1e293b; color: #93c5fd; padding: 1px 6px; border-radius: 5px; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 6px 0; }
.kpi-box {
    background: #0f172a; border: 1px solid #334155; border-radius: 10px;
    padding: 10px 10px; text-align: center;
}
.kpi-val { font-size: 1.15rem; font-weight: 700; color: #f8fafc; }
.kpi-lbl { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
.verdict-clean { color: #4ade80; font-weight: 700; }
.verdict-off { color: #f87171; font-weight: 700; }
.bar-track {
    position: relative; height: 10px; background: #1e293b; border-radius: 999px;
    overflow: hidden; margin: 8px 0 4px 0; border: 1px solid #334155;
}
.bar-fill-clean { height: 100%; background: linear-gradient(90deg, #22c55e, #4ade80); }
.bar-fill-off { height: 100%; background: linear-gradient(90deg, #ef4444, #f97316); }
.bar-thr { position: absolute; top: 0; bottom: 0; width: 2px; background: #facc15; }
"""

def render_hero(engine: ModerationEngine) -> str:
    n_q = engine.mbert_int8.quantized_layers
    eng = engine.mbert_int8.engine
    return f"""
<div class="mod-hero">
  <h1>🛡️ Real-Time Chat Moderation — Live CPU Demo</h1>
  <p>SWE015 · Use Case 4 · Classify chat messages live as <b>CLEAN</b> or <b>OFFENSIVE</b>, automatically mask flagged text, log per-message latency, and switch across all 5 models.</p>
  <div class="pill-row">
    <span class="pill pill-green">Default: Dynamic INT8 mBERT ({n_q} Linear Layers · {html.escape(str(eng))})</span>
    <span class="pill">CPU-Only Inference (batch=1)</span>
    <span class="pill pill-amber">Latency Budget: ≤ {LATENCY_BUDGET_MS:.0f} ms / msg</span>
    <span class="pill">Jigsaw 50k · 5 Models</span>
  </div>
</div>
"""


def render_model_card(info: dict, model_name: str) -> str:
    m = info[model_name]
    badge_cls = "pill-green" if "INT8" in m["precision"] else "pill"
    return (
        f'<div class="mod-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;">'
        f'<b style="font-size:0.96rem;color:#fff;">{html.escape(model_name)}</b>'
        f'<span class="pill {badge_cls}">{html.escape(m["precision"])}</span>'
        f'</div>'
        f'<div><b>Technique Area:</b> {html.escape(m["area"])}</div>'
        f'<div><b>Preprocessing:</b> <code>{html.escape(m["preprocess"])}</code></div>'
        f'<div><b>Threshold:</b> <code>{m["threshold"]}</code> · <b>Runtime:</b> {html.escape(m["details"])}</div>'
        f'</div>'
    )


def render_kpis(log_rows: list) -> str:
    n = len(log_rows)
    flagged = sum(1 for r in log_rows if r[2] == "OFFENSIVE")
    last_ms = f"{float(log_rows[-1][4]):.1f} ms" if n else "—"
    mean_ms = f"{sum(float(r[4]) for r in log_rows) / n:.1f} ms" if n else "—"
    last_color = "#f8fafc"
    if n:
        last_color = "#4ade80" if float(log_rows[-1][4]) <= LATENCY_BUDGET_MS else "#fbbf24"
    return (
        '<div class="kpi-grid">'
        f'<div class="kpi-box"><div class="kpi-val">{n}</div><div class="kpi-lbl">Messages</div></div>'
        f'<div class="kpi-box"><div class="kpi-val" style="color:#f87171;">{flagged}</div><div class="kpi-lbl">Masked</div></div>'
        f'<div class="kpi-box"><div class="kpi-val" style="color:{last_color};">{last_ms}</div><div class="kpi-lbl">Last Latency</div></div>'
        f'<div class="kpi-box"><div class="kpi-val">{mean_ms}</div><div class="kpi-lbl">Session Mean</div></div>'
        '</div>'
    )


def render_inspector(res: dict | None) -> str:
    if not res:
        return (
            '<div class="mod-card" style="color:#94a3b8;">'
            'Send a message to inspect its toxicity score, threshold bar, normalized text, and CPU latency.'
            '</div>'
        )
    is_off = res["is_offensive"]
    v_cls = "verdict-off" if is_off else "verdict-clean"
    fill_cls = "bar-fill-off" if is_off else "bar-fill-clean"
    p_pct = max(0.0, min(100.0, res["p_offensive"] * 100.0))
    thr_pct = max(0.0, min(100.0, res["threshold"] * 100.0))
    bud_cls = "pill-green" if res["within_budget"] else "pill-amber"
    bud_txt = f"{res['latency_ms']:.2f} ms · ≤{LATENCY_BUDGET_MS:.0f} ms ✓" if res["within_budget"] else f"{res['latency_ms']:.2f} ms · >{LATENCY_BUDGET_MS:.0f} ms"
    norm_txt = html.escape(res["normalized"][:140])
    return (
        '<div class="mod-card">'
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">'
        f'<span class="{v_cls}" style="font-size:1.02rem;">'
        f'{"🚫 OFFENSIVE (MASKED)" if is_off else "✅ CLEAN"} · {res["confidence"]*100:.1f}% confidence</span>'
        f'<span class="pill {bud_cls}">⏱️ {bud_txt}</span>'
        '</div>'
        f'<div class="bar-track">'
        f'<div class="{fill_cls}" style="width:{p_pct:.1f}%;"></div>'
        f'<div class="bar-thr" style="left:{thr_pct:.1f}%;" title="Threshold {res["threshold"]:.2f}"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">'
        f'<span>P(offensive) = <b>{res["p_offensive"]:.3f}</b></span>'
        f'<span>Decision threshold (yellow line) = <b>{res["threshold"]:.2f}</b></span>'
        f'</div>'
        f'<div style="font-size:0.82rem;color:#cbd5e1;"><b>Normalized input:</b> <code>{norm_txt or "(empty)"}</code></div>'
        '</div>'
    )


def build_ui(engine: ModerationEngine) -> gr.Blocks:
    def on_model_change(model_name: str) -> str:
        return render_model_card(engine.info, model_name)

    def send_message(message: str, model_name: str, chat_history: list, log_rows: list, last_msg: str):
        text = (message or "").strip()
        if not text:
            return ("", chat_history, chat_history, log_rows, log_rows,
                    render_kpis(log_rows), gr.update(), last_msg)

        res = engine.moderate(text, model_name=model_name)

        if res["is_offensive"]:
            user_bubble = f"🚫 `[MESSAGE MASKED]` {res['display_text']}"
            mod_bubble = (
                f"🔴 **OFFENSIVE** · Confidence **{res['confidence']*100:.1f}%** "
                f"(`P(off)={res['p_offensive']:.3f}`, thr `{res['threshold']:.2f}`)  \n"
                f"⏱️ `{res['latency_ms']:.2f} ms` · `{model_name.split(' (')[0]}`"
            )
        else:
            user_bubble = res["display_text"]
            mod_bubble = (
                f"🟢 **CLEAN** · Confidence **{res['confidence']*100:.1f}%** "
                f"(`P(off)={res['p_offensive']:.3f}`, thr `{res['threshold']:.2f}`)  \n"
                f"⏱️ `{res['latency_ms']:.2f} ms` · `{model_name.split(' (')[0]}`"
            )

        new_history = list(chat_history) + [
            {"role": "user", "content": user_bubble},
            {"role": "assistant", "content": mod_bubble},
        ]
        short_msg = "[MASKED]" if res["is_offensive"] else (text if len(text) <= 26 else text[:23] + "...")
        new_logs = list(log_rows) + [[
            len(log_rows) + 1,
            model_name.split(" (")[0],
            res["label"],
            f"{res['confidence']*100:.1f}%",
            res["latency_ms"],
            "Yes" if res["within_budget"] else "No",
            short_msg,
        ]]
        return ("", new_history, new_history, new_logs, new_logs,
                render_kpis(new_logs), render_inspector(res), text)

    def compare_models(message: str, last_msg: str):
        text = (message or "").strip() or (last_msg or "").strip()
        if not text:
            return []
        return engine.compare_all(text)

    def clear_all():
        return [], [], [], [], render_kpis([]), render_inspector(None), [], ""

    # Gradio 5 requires type="messages"; Gradio 6 removed 'type' (messages is the only format)
    chatbot_kwargs = {"label": "Live Gaming Chat Room", "height": 390}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        chatbot_kwargs["type"] = "messages"

    blocks_kwargs = {"title": "Real-Time Chat Moderation (SWE015)"}
    if "css" not in inspect.signature(gr.Blocks.launch).parameters:
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(render_hero(engine))

        with gr.Row(equal_height=False):
            # Left column: Chat room + live input + quick examples
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(**chatbot_kwargs)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type a live chat message and press Enter...",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    compare_btn = gr.Button("⚖️ Compare All 5 Models (input or last message)", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear Chat")

                gr.Examples(
                    label="Quick sample messages (demo also supports typing any new message live)",
                    examples=[
                        ["Good game everyone, that was a close match!"],
                        ["y0u are a stuuuupid id10t!!!"],
                        ["You are not stupid at all, nice play!"],
                        ["Hey donkey, its your uncle."],
                        ["Bu maç çok zordu ama teşekkürler arkadaşlar!"],
                    ],
                    inputs=msg_box,
                )

            # Right column: Model switcher + live inspector + KPIs + latency log
            with gr.Column(scale=2):
                model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Active Moderation Model (switch live)",
                )
                model_card = gr.HTML(render_model_card(engine.info, DEFAULT_MODEL))
                inspector_html = gr.HTML(render_inspector(None))
                kpi_html = gr.HTML(render_kpis([]))
                log_table = gr.Dataframe(
                    headers=["#", "Model", "Label", "Conf.", "ms", "≤100ms", "Message"],
                    datatype=["number", "str", "str", "str", "number", "str", "str"],
                    value=[],
                    label="Per-Message Response-Time Log",
                    interactive=False,
                )

        with gr.Accordion("⚖️ Side-by-Side Comparison of All 5 Models", open=True):
            compare_table = gr.Dataframe(
                headers=["Model", "Label", "Confidence", "P(offensive)", "Latency (ms)", "≤100ms"],
                datatype=["str", "str", "str", "str", "str", "str"],
                value=[],
                interactive=False,
            )

        chat_state = gr.State([])
        log_state = gr.State([])
        last_msg_state = gr.State("")

        model_dd.change(on_model_change, inputs=model_dd, outputs=model_card)

        send_inputs = [msg_box, model_dd, chat_state, log_state, last_msg_state]
        send_outputs = [
            msg_box, chat_state, chatbot, log_state, log_table,
            kpi_html, inspector_html, last_msg_state,
        ]
        send_btn.click(send_message, inputs=send_inputs, outputs=send_outputs)
        msg_box.submit(send_message, inputs=send_inputs, outputs=send_outputs)

        compare_btn.click(compare_models, inputs=[msg_box, last_msg_state], outputs=compare_table)
        clear_btn.click(
            clear_all,
            outputs=[
                chat_state, chatbot, log_state, log_table,
                kpi_html, inspector_html, compare_table, last_msg_state,
            ],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (for Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading moderation models on CPU (downloading from HF Hub on first run)...", flush=True)
    engine = ModerationEngine()
    print(f"Ready! Default model: {DEFAULT_MODEL} | {engine.info[DEFAULT_MODEL]['details']}", flush=True)

    app = build_ui(engine)
    launch_kwargs = {
        "server_name": "0.0.0.0" if args.share else "127.0.0.1",
        "server_port": args.port,
        "share": args.share,
    }
    if "css" in inspect.signature(gr.Blocks.launch).parameters:
        launch_kwargs["css"] = CUSTOM_CSS
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
