"""MLM encoder 가중치를 옮기되 원본 KR-FinBERT pooler는 유지해 공정한 Model B 시작점을 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="snunlp/KR-FinBert-SC")
    parser.add_argument("--mlm", type=Path, default=Path("artifacts/finbert/model_b/mlm/best"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/finbert/model_b/mlm/encoder_best"))
    args = parser.parse_args()

    encoder = AutoModel.from_pretrained(args.base_model)
    mlm = AutoModelForMaskedLM.from_pretrained(args.mlm)
    result = encoder.load_state_dict(mlm.base_model.state_dict(), strict=False)
    allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    missing = set(result.missing_keys)
    if missing != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            f"예상하지 못한 encoder 변환 결과: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.mlm).save_pretrained(args.output)
    (args.output / "export_report.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "mlm_checkpoint": str(args.mlm),
                "mlm_encoder_loaded": True,
                "original_pooler_preserved": True,
                "missing_keys": result.missing_keys,
                "unexpected_keys": result.unexpected_keys,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
