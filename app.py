"""Interactive Real-Time Chat Moderation App (Gradio) — SWE015 Use Case 4.

Run locally : python app.py
Run in Colab: python app.py --share
"""
import argparse
import inspect

import gradio as gr

from src.moderation import DEFAULT_MODEL, LATENCY_BUDGET_MS, MODEL_CHOICES, ModerationEngine

print("Loading moderation models on CPU (downloading from HF Hub on first run)...", flush=True)
ENGINE = ModerationEngine()
print(f"Ready! Default model: {DEFAULT_MODEL} | {ENGINE.info[DEFAULT_MODEL]['details']}", flush=True)


def format_model_card(model_name: str) -> str:
    m = ENGINE.info[model_name]
    return (
        f"**Active Model:** `{model_name}`  \n"
        f"- **Technique Area:** {m['area']}  \n"
        f"- **Precision:** `{m['precision']}`  \n"
        f"- **Preprocessing:** {m['preprocess']}  \n"
        f"- **Decision Threshold:** `{m['threshold']}`  \n"
        f"- **Runtime:** {m['details']}  \n"
        f"- **Latency Budget:** ≤ {LATENCY_BUDGET_MS:.0f} ms / message (CPU)"
    )


def _latency_summary(log_rows: list) -> str:
    if not log_rows:
        return "No messages sent yet."
    lats = [float(r[4]) for r in log_rows]
    flagged = sum(1 for r in log_rows if r[2] == "OFFENSIVE")
    return (f"**Session stats:** {len(log_rows)} messages · {flagged} flagged & masked · "
            f"mean latency **{sum(lats)/len(lats):.2f} ms** · last **{lats[-1]:.2f} ms**")


def send_message(message: str, model_name: str, chat_history: list, log_rows: list, last_msg: str):
    text = (message or "").strip()
    if not text:
        return "", chat_history, chat_history, log_rows, log_rows, _latency_summary(log_rows), last_msg

    res = ENGINE.moderate(text, model_name=model_name)

    if res["is_offensive"]:
        user_bubble = f"🚫 `[MASKED]` {res['display_text']}"
        mod_bubble = (
            f"🔴 **OFFENSIVE** · Confidence: **{res['confidence']*100:.1f}%** "
            f"(P(off)={res['p_offensive']:.3f}, thr={res['threshold']:.2f})  \n"
            f"⏱️ `{res['latency_ms']:.2f} ms` · Model: `{model_name}`"
        )
    else:
        user_bubble = res["display_text"]
        mod_bubble = (
            f"🟢 **CLEAN** · Confidence: **{res['confidence']*100:.1f}%** "
            f"(P(off)={res['p_offensive']:.3f}, thr={res['threshold']:.2f})  \n"
            f"⏱️ `{res['latency_ms']:.2f} ms` · Model: `{model_name}`"
        )

    new_history = list(chat_history) + [
        {"role": "user", "content": user_bubble},
        {"role": "assistant", "content": mod_bubble},
    ]
    short_msg = "[MASKED]" if res["is_offensive"] else (text if len(text) <= 28 else text[:25] + "...")
    new_logs = list(log_rows) + [[
        len(log_rows) + 1,
        model_name.split(" (")[0],
        res["label"],
        f"{res['confidence']*100:.1f}%",
        res["latency_ms"],
        "Yes" if res["within_budget"] else "No",
        short_msg,
    ]]
    return "", new_history, new_history, new_logs, new_logs, _latency_summary(new_logs), text


def compare_models(message: str, last_msg: str):
    text = (message or "").strip() or (last_msg or "").strip()
    if not text:
        return []
    return ENGINE.compare_all(text)


def clear_all():
    return [], [], [], [], "No messages sent yet.", [], ""


def build_ui() -> gr.Blocks:
    # Compatible with both Gradio 5 (needs type="messages") and Gradio 6+ (messages is default)
    chatbot_kwargs = {"label": "Live Moderation Chat", "height": 400}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        chatbot_kwargs["type"] = "messages"

    with gr.Blocks(title="Real-Time Chat Moderation (SWE015)") as demo:
        gr.Markdown(
            "# Real-Time Chat Moderation (SWE015 – Use Case 4)\n"
            "Type a chat message below to classify it live as **CLEAN** or **OFFENSIVE** on CPU. "
            "Offensive messages are automatically masked. Switch models or compare all 5 models in the side panel."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(**chatbot_kwargs)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type a chat message (e.g. 'gg wp everyone!' or 'y0u are so000 stup1d!!!')...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    compare_btn = gr.Button("Compare All 5 Models (input or last message)")
                    clear_btn = gr.Button("Clear Chat")

                gr.Examples(
                    label="Quick test messages (you can also type anything live)",
                    examples=[
                        ["Good game everyone, that was a close match!"],
                        ["y0u are so000 stup1d and usel3ss!!!"],
                        ["HE IS NOT A PORN PRODUCER"],
                        ["Hey donkey, its your uncle."],
                        ["Bu maç çok zordu ama teşekkürler arkadaşlar!"],
                    ],
                    inputs=msg_box,
                )

            with gr.Column(scale=2):
                model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Select Moderation Model",
                )
                model_card = gr.Markdown(format_model_card(DEFAULT_MODEL))
                stats_md = gr.Markdown("No messages sent yet.")
                log_table = gr.Dataframe(
                    headers=["#", "Model", "Label", "Conf.", "Latency (ms)", "≤100ms", "Message"],
                    datatype=["number", "str", "str", "str", "number", "str", "str"],
                    value=[],
                    label="Per-Message Response-Time Log",
                    interactive=False,
                )

        with gr.Accordion("Side-by-Side Comparison of All 5 Models", open=True):
            compare_table = gr.Dataframe(
                headers=["Model", "Label", "Confidence", "P(offensive)", "Latency (ms)", "≤100ms"],
                datatype=["str", "str", "str", "str", "str", "str"],
                value=[],
                interactive=False,
            )

        chat_state = gr.State([])
        log_state = gr.State([])
        last_msg_state = gr.State("")

        model_dd.change(format_model_card, inputs=model_dd, outputs=model_card)

        send_inputs = [msg_box, model_dd, chat_state, log_state, last_msg_state]
        send_outputs = [msg_box, chat_state, chatbot, log_state, log_table, stats_md, last_msg_state]
        send_btn.click(send_message, inputs=send_inputs, outputs=send_outputs)
        msg_box.submit(send_message, inputs=send_inputs, outputs=send_outputs)

        compare_btn.click(compare_models, inputs=[msg_box, last_msg_state], outputs=compare_table)
        clear_btn.click(
            clear_all,
            outputs=[chat_state, chatbot, log_state, log_table, stats_md, compare_table, last_msg_state],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (for Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_ui()
    app.launch(server_name="0.0.0.0" if args.share else "127.0.0.1",
               server_port=args.port, share=args.share)
