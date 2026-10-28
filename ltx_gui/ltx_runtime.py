"""LTX-2.5 native model runtime (Lightricks official components; no Jasna).

The upstream ltx-core/ltx-pipelines package is distributed under its own
LTX-2.x license. This adapter is project-authored, imports those modules only
when selected, and never downloads/redistributes weights or replaces them
with a guessed mean. Retake is temporal regeneration, not spatial inpainting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

from .weights import LTX25ModelPack


@dataclass(frozen=True)
class RetakeRequest:
    video: Path
    output: Path
    weights_root: Path
    prompt: str
    start_time: float
    end_time: float
    seed: int = 20260923

    def validate(self) -> None:
        if not self.video.is_file():
            raise FileNotFoundError(self.video)
        if self.output.exists() or self.output.resolve() == self.video.resolve():
            raise FileExistsError("입력 파일 덮어쓰기 및 기존 출력 덮어쓰기를 허용하지 않습니다.")
        if not self.prompt.strip():
            raise ValueError("LTX-2.5 Retake에는 텍스트 프롬프트가 필요합니다.")
        if not 0 <= self.start_time < self.end_time or self.seed < 0 or self.seed >= 2**63:
            raise ValueError("시각 범위 또는 시드가 잘못되었습니다.")


def run_temporal_retake(req: RetakeRequest) -> Path:
    """Run actual DiT+VAE, using *real* LTX-2.5 pretrained split checkpoints.

    Output is a time-range retake of the complete video, not pixel-wise masked
    restoration. Its content is generated and may not match original ground truth.
    Requires current upstream dependencies and CUDA capable hardware.
    """
    req.validate()
    pack=LTX25ModelPack.from_directory(req.weights_root, distilled=True)
    pack.verify()
    try:
        import torch
        from ltx_pipelines.retake import RetakePipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
        from ltx_core.model.video_vae import get_video_chunks_number
    except ImportError as exc:
        raise RuntimeError("LTX 공식 ltx-core/ltx-pipelines를 별도로 설치해야 합니다.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. CPU 가중치 평균으로 대체하지 않습니다.")
    metadata=get_videostream_metadata(str(req.video))
    if (metadata.frames-1)%8 or metadata.width%32 or metadata.height%32:
        raise ValueError("입력은 8k+1 프레임, 가로·세로는 32의 배수여야 합니다.")
    if req.end_time>metadata.frames/metadata.fps:
        raise ValueError("Retake 종료 시각이 영상 길이보다 깁니다.")
    paths=ModelPaths.from_split(transformer_path=str(pack.transformer),
                                text_encoder_path=str(pack.text_encoder),
                                video_vae_path=str(pack.video_vae),
                                audio_vae_path=str(pack.audio_vae))
    pipeline=RetakePipeline(model_paths=paths,loras=[],distilled=True,device=torch.device("cuda"))
    req.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=req.output.parent, prefix=".ltx25_staging_") as directory:
        staged = Path(directory) / ("result" + req.output.suffix)
        with torch.inference_mode():
            result=pipeline(video_path=str(req.video),prompt=req.prompt,
                            start_time=req.start_time,end_time=req.end_time,
                            seed=req.seed,regenerate_video=True,regenerate_audio=False)
            encode_video(video=result.video,fps=int(metadata.fps),audio=result.audio,
                         output_path=str(staged),
                         video_chunks_number=get_video_chunks_number(result.num_frames,result.tiling_config))
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError("LTX 추론 출력 파일이 생성되지 않았습니다.")
        # Exclusive publication; never overwrite data created during model inference.
        os.link(staged,req.output)
    return req.output
