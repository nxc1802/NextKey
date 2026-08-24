#!/usr/bin/env python3
"""Run multi-level benchmark (Character, Word, Sentence) across all Phase 1, Phase 2, and Phase 3 models."""

import json
from pathlib import Path
import torch
from nextkey.data.dataset import find_dataset_root, load_examples
from nextkey.data.tokenizer import CharVocab
from nextkey.engine.evaluator import ModelEvaluator
from nextkey.engine.quantization import apply_dynamic_quantization, QuantizedBiGRUCharTagger
from nextkey.models.base import create_model
import nextkey.models


def load_model(ckpt_path: Path, device: torch.device):
    vocab_path = ckpt_path.parent / "vocab.json"
    vocab = CharVocab.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("config", {})
    if "model" in model_cfg:
        model_cfg = model_cfg["model"]
    elif "model_config" in ckpt:
        model_cfg = ckpt["model_config"]

    model = create_model(
        model_cfg.get("name", "bigru"),
        vocab_size=vocab.input_vocab_size,
        num_target_classes=vocab.target_vocab_size,
        embed_dim=model_cfg.get("embed_dim", 32),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 1),
        dropout=0.0,
    )
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    model.load_state_dict(state)
    model.eval()
    return model, vocab


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset_root = find_dataset_root()

    in_domain_files = sorted((dataset_root / "test" / "in_domain").glob("*.jsonl"))
    in_domain_examples = []
    for f in in_domain_files:
        in_domain_examples.extend(load_examples(f, max_samples=1500))

    models_to_eval = [
        # Phase 1
        ("P1: BiGRU Baseline", Path("artifacts 2/phase2/depth_1/best_model.pt"), "FP32"),
        # Phase 2
        ("P2: Topo-A Wide/Shallow", Path("artifacts 2/phase2/topo_a_wide_shallow/best_model.pt"), "FP32"),
        ("P2: Width-XL", Path("artifacts 2/phase2/bigru-e48-h96-L1/best_model.pt"), "FP32"),
        ("P2: Width-XS", Path("artifacts 2/phase2/width_xs/best_model.pt"), "FP32"),
        ("P2: Width-XXS", Path("artifacts 2/phase2/width_xxs/best_model.pt"), "FP32"),
        ("P2: Width-XXXS", Path("artifacts 2/phase2/width_xxxs/best_model.pt"), "FP32"),
        # Phase 3
        ("P3: Student PTQ Only (No KD)", Path("artifacts/phase3/quantization_only/student_ptq_only.pt"), "PTQ_INT8"),
        ("P3: Student Trad KD FP32", Path("artifacts 2/phase3/traditional_kd/best_model.pt"), "FP32"),
        ("P3: Student Trad KD + PTQ", Path("artifacts/phase3/quantization_only/student_trad_kd_ptq_int8.pt"), "PTQ_INT8"),
        ("P3: Student QKD INT8", Path("artifacts 2/phase3/qkd_int8/best_model.pt"), "QKD_INT8"),
    ]

    all_results = []

    for name, ckpt_path, fmt in models_to_eval:
        if not ckpt_path.exists():
            continue
        print(f"🔬 Evaluating: {name} ({fmt})...")
        if fmt == "FP32":
            model, vocab = load_model(ckpt_path, device)
            evaluator = ModelEvaluator(model, vocab, device=device, batch_size=256)
            res, _ = evaluator.evaluate_examples(in_domain_examples)
        elif fmt == "PTQ_INT8":
            base_model, vocab = load_model(Path("artifacts 2/phase2/width_xs/best_model.pt"), torch.device("cpu"))
            ptq_model = apply_dynamic_quantization(base_model)
            evaluator = ModelEvaluator(ptq_model, vocab, device=torch.device("cpu"), batch_size=256)
            res, _ = evaluator.evaluate_examples(in_domain_examples)
        elif fmt == "QKD_INT8":
            qkd_vocab = CharVocab.load(ckpt_path.parent / "vocab.json")
            qkd_model = QuantizedBiGRUCharTagger(
                vocab_size=qkd_vocab.input_vocab_size,
                num_target_classes=qkd_vocab.target_vocab_size,
                embed_dim=32,
                hidden_dim=64,
                num_layers=1,
                dropout=0.0,
            )
            qkd_state = torch.load(ckpt_path, map_location="cpu")
            qkd_model.load_state_dict(qkd_state.get("state_dict") or qkd_state.get("model_state_dict"))
            evaluator = ModelEvaluator(qkd_model, qkd_vocab, device=torch.device("cpu"), batch_size=256)
            res, _ = evaluator.evaluate_examples(in_domain_examples)

        res["model_name"] = name
        res["format"] = fmt
        all_results.append(res)
        print(f"   ✓ Clean CER: {res['corpus_cer']*100:.2f}% | WER: {res['corpus_wer']*100:.2f}% | Word F1: {res['word_f1']*100:.2f}% | Near-Perf (<=5%): {res['sentence_near_perfect_5pct']*100:.2f}% | BLEU-4: {res['bleu_4']} | ROUGE-L: {res['rouge_l_f1']}")

    out_file = Path("artifacts/all_phases_multilevel_benchmark.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 Saved all results to: {out_file}")


if __name__ == "__main__":
    main()
