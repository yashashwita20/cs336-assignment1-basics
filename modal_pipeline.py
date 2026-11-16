import pickle
from pathlib import Path

import modal

app = modal.App("cs336-bpe-tokenizer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "regex>=2026.3.32",
        "tqdm>=4.67",
    )
    .add_local_python_source("cs336_basics")
)

volume = modal.Volume.from_name("cs336-tokenizer-vol", create_if_missing=True)
VOLUME_PATH = Path("/vol")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 3,  # 3 hours
)
def train(data_filename: str, vocab_size: int, special_tokens: list[str], output_name: str):
    from cs336_basics.tokenizer import train_bpe

    input_path = VOLUME_PATH / "data" / data_filename
    print(f"Training BPE on {input_path} | vocab_size={vocab_size}")

    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    out_path = VOLUME_PATH / "outputs" / f"{output_name}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    volume.commit()
    print(f"Saved to {out_path}")
    return str(out_path)


@app.local_entrypoint()
def main(
    input_path: str = "data/TinyStoriesV2-GPT4-train.txt",
    vocab_size: int = 10000,
    special_tokens: str = "<|endoftext|>",
    output_name: str = "",
):
    """
    Run BPE tokenizer training on Modal.

    Examples:
        modal run modal_pipeline.py
        modal run modal_pipeline.py --input-path data/owt_train.txt --vocab-size 32000
        modal run modal_pipeline.py --input-path data/owt_train.txt --vocab-size 32000 --output-name owt_32k
        modal run modal_pipeline.py --special-tokens "<|endoftext|>,<|padding|>"
    """
    local_path = Path(input_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    filename = local_path.name
    if not output_name:
        output_name = f"{local_path.stem}_vocab{vocab_size}"

    tokens = [t.strip() for t in special_tokens.split(",") if t.strip()]

    print(f"Uploading {local_path} to volume...")
    with volume.batch_upload() as batch:
        batch.put_file(local_path, f"data/{filename}")

    train.remote(
        data_filename=filename,
        vocab_size=vocab_size,
        special_tokens=tokens,
        output_name=output_name,
    )
