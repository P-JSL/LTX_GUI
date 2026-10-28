"""Model checkpoint resolution and **pretrained** weight policy.

This project deliberately refuses to invent tensor values from a statistical mean.
Weight-space averages of compatible trained checkpoints are an *experiment*, not a
replacement for original LTX-2.5 weights. The original downloads remain external.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OFFICIAL_LTX25 = "https://huggingface.co/Lightricks/LTX-2.5"


@dataclass(frozen=True)
class LTX25ModelPack:
    transformer: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path

    @classmethod
    def from_directory(cls, root: str | Path, *, distilled: bool = True,
                       diffusion_vae: bool = False) -> "LTX25ModelPack":
        """Select official split-file layout without guessing other model families."""
        root = Path(root).expanduser().resolve()
        return cls(
            transformer=root / "diffusion_models" / (
                "ltx-2.5-22b-distilled-transformer-bf16.safetensors" if distilled
                else "ltx-2.5-22b-dev-transformer-bf16.safetensors"),
            text_encoder=root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            video_vae=root / "vae" / (
                "ltx-2.5-video-vae-bf16.safetensors" if diffusion_vae
                else "ltx-2.5-video-vae-conv-bf16.safetensors"),
            audio_vae=root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
        )

    def verify(self, *, check_safetensors_header: bool = True) -> None:
        """Fail closed on absent/corrupt checkpoint headers; never substitute averages.

        Does NOT prove tensor correspondence, legal rights, or inference usability.
        """
        missing = [str(p) for p in (self.transformer, self.text_encoder,
                                     self.video_vae, self.audio_vae) if not p.is_file()]
        if missing:
            raise FileNotFoundError("LTX-2.5 학습 가중치 누락: " + "; ".join(missing))
        if not check_safetensors_header:
            return
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors 설치가 필요합니다.") from exc
        for path in (self.transformer, self.text_encoder, self.video_vae, self.audio_vae):
            # safe_open validates the metadata without eagerly loading the 22B parameters.
            with safe_open(path, framework="pt", device="cpu") as handle:
                if not tuple(handle.keys()):
                    raise ValueError(f"빈 모델 체크포인트: {path}")


def validate_weight_mix(checkpoint_paths: list[str | Path], coefficients: list[float]) -> None:
    """Validate a *research-only* merge request; does not claim aligned weights.

    Compatible architecture, identical tensor names/shapes, shared training basin,
    and measured quality are additionally necessary for meaningful weight soups.
    """
    import math
    if len(checkpoint_paths) < 2 or len(checkpoint_paths) != len(coefficients):
        raise ValueError("최소 두 개의 실제 학습 체크포인트와 각각의 가중 계수가 필요합니다.")
    if len({str(Path(p).resolve()) for p in checkpoint_paths}) != len(checkpoint_paths):
        raise ValueError("중복된 체크포인트는 평균 대상이 아닙니다.")
    if any(not Path(p).is_file() for p in checkpoint_paths):
        raise FileNotFoundError("모든 학습 체크포인트가 존재해야 합니다.")
    if any(not math.isfinite(c) or c < 0 for c in coefficients) or not math.isclose(sum(coefficients), 1.0, abs_tol=1e-6):
        raise ValueError("평균 계수는 0 이상이고 합계가 1이어야 합니다.")
