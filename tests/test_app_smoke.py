"""Smoke test for the full ModerationEngine (all 5 models, masking, INT8 verification).

Run: python tests/test_app_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.moderation import DEFAULT_MODEL, MODEL_CHOICES, ModerationEngine, mask_message  # noqa: E402

assert mask_message("you are stupid!") == "*** *** ******!"
print("OK  mask_message")

engine = ModerationEngine()
assert engine.mbert_int8.quantized_layers == 74, engine.mbert_int8.quantized_layers
print(f"OK  default model is {DEFAULT_MODEL} with {engine.mbert_int8.quantized_layers} quantized Linear layers "
      f"(engine={engine.mbert_int8.engine})")

samples = [
    "Good game everyone, well played!",
    "y0u are a stuuuupid id10t!!!",
    "Bu oyun çok güzeldi, teşekkürler!",
]
for s in samples:
    print(f"\nInput: {s!r}")
    for m in MODEL_CHOICES:
        r = engine.moderate(s, model_name=m)
        if r["is_offensive"]:
            assert "*" in r["display_text"], r
        else:
            assert r["display_text"] == s, r
        print(f"  {m:48} -> {r['label']:9} conf={r['confidence']:.2f} "
              f"lat={r['latency_ms']:6.2f} ms | {r['display_text']!r}")

print("\nALL APP SMOKE TESTS PASSED")
