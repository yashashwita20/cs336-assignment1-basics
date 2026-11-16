"""Quick smoke test — trains on a tiny in-memory string to verify Modal + tokenizer wiring."""
import modal

app = modal.App("cs336-bpe-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex>=2026.3.32", "tqdm>=4.67")
    .add_local_python_source("cs336_basics")
)


@app.function(image=image, cpu=2, timeout=120)
def smoke_test():
    import tempfile
    from cs336_basics.tokenizer import train_bpe

    text = "hello world " * 500 + "foo bar baz " * 300
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp_path = f.name

    vocab, merges = train_bpe(
        input_path=tmp_path,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    print(f"OK — vocab size: {len(vocab)}, merges: {len(merges)}")
    return len(vocab), len(merges)


@app.local_entrypoint()
def main():
    vocab_size, n_merges = smoke_test.remote()
    print(f"Smoke test passed: vocab={vocab_size}, merges={n_merges}")
