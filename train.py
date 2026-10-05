"""Reproducible CPU/GPU training, validation selection, one final test evaluation."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from model import ByteLM, Config

ROOT = Path(__file__).resolve().parent


def data(split):
    return torch.tensor(np.frombuffer((ROOT / "data" / f"{split}.bin").read_bytes(), dtype=np.uint8).copy(), dtype=torch.long)


def batch(tokens, context, size, rng, device):
    starts = torch.randint(len(tokens) - context, (size,), generator=rng)
    ix = starts[:, None] + torch.arange(context + 1)
    z = tokens[ix].to(device)
    return z[:, :-1], z[:, 1:]


@torch.inference_mode()
def evaluate(model, tokens, context=None, max_windows=None):
    model.eval()
    context = context or model.config.context
    # Non-overlapping targets, with deterministic starts; partial tail excluded and reported.
    starts = torch.arange(0, len(tokens) - context, context)
    if max_windows is not None and len(starts) > max_windows:
        starts = starts[torch.linspace(0, len(starts) - 1, max_windows).long()]
    total, count = 0.0, 0
    losses = []
    device = next(model.parameters()).device
    for offset in range(0, len(starts), 16):
        ix = starts[offset:offset + 16, None] + torch.arange(context + 1)
        z = tokens[ix].to(device)
        logits = model(z[:, :-1])
        per_token = F.cross_entropy(logits.reshape(-1, 256), z[:, 1:].reshape(-1), reduction="none").reshape(len(z), context)
        losses.extend(per_token.mean(1).cpu().tolist())
        total += per_token.sum().item()
        count += per_token.numel()
    mean = total / count
    return {"nll_nats": mean, "bits_per_byte": mean / math.log(2), "byte_perplexity": math.exp(mean), "evaluated_bytes": count, "window_losses": losses}


def bigram(train_tokens, test_tokens):
    counts = torch.bincount(train_tokens[:-1] * 256 + train_tokens[1:], minlength=256 * 256).double().reshape(256, 256) + .1
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[test_tokens[:-1], test_tokens[1:]].log().mean().item()
    return {"nll_nats": loss, "bits_per_byte": loss / math.log(2), "byte_perplexity": math.exp(loss), "smoothing": .1, "evaluated_bytes": len(test_tokens) - 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="rope-seed42")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(rope=not args.no_rope)
    model = ByteLM(config).to(device)
    train_tokens, val_tokens, test_tokens = [data(s) for s in ("train", "validation", "test")]
    rng = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.1, betas=(.9, .95))
    directory = ROOT / "runs" / args.name
    if (directory / "metrics.json").exists():
        raise FileExistsError("Finished run exists; choose a new name")
    directory.mkdir(parents=True, exist_ok=True)
    trace = []
    best, best_step, best_state = float("inf"), 0, None
    begin = time.perf_counter()
    print(f"{args.name}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}", flush=True)
    for step in range(args.steps + 1):
        if step % 250 == 0 or step == args.steps:
            val = evaluate(model, val_tokens, max_windows=128)
            row = {"step": step, "validation_bpb": val["bits_per_byte"], "elapsed_seconds": time.perf_counter() - begin}
            trace.append(row)
            print(json.dumps(row), flush=True)
            if val["nll_nats"] < best:
                best, best_step = val["nll_nats"], step
                best_state = copy.deepcopy(model.state_dict())
            (directory / "history.json").write_text(json.dumps(trace, indent=2))
        if step == args.steps:
            break
        model.train()
        warmup = min(1.0, (step + 1) / 100)
        factor = .1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * warmup * factor
        x, y = batch(train_tokens, config.context, args.batch_size, rng, device)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.load_state_dict(best_state)
    model.save_pretrained(directory)
    final = {"name": args.name, "seed": args.seed, "steps": args.steps, "selected_step": best_step, "parameters": sum(p.numel() for p in model.parameters()), "device": device, "torch_version": torch.__version__, "threads": args.threads, "batch_size": args.batch_size, "training_bytes_presented": args.steps * config.context * args.batch_size, "elapsed_seconds": time.perf_counter() - begin, "validation": evaluate(model, val_tokens), "test": evaluate(model, test_tokens), "bigram_test": bigram(train_tokens, test_tokens), "config": config.__dict__}
    # Context analysis is descriptive, not used to select a model using the test set.
    final["validation_context"] = {str(c): evaluate(model, val_tokens, context=c, max_windows=128)["bits_per_byte"] for c in (16, 32, 64, 128)}
    (directory / "metrics.json").write_text(json.dumps(final, indent=2))
    samples = {prompt: model.generate(prompt, seed=args.seed) for prompt in ("ROMEO:\n", "First Citizen:\n", "To be, or not to be")}
    (directory / "samples.json").write_text(json.dumps(samples, indent=2))
    print("FINISHED", args.name, "test bpb", final["test"]["bits_per_byte"], flush=True)


if __name__ == "__main__":
    main()
