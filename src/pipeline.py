import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)
from pipelines.models import TextToImageRequest
from torch import Generator


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-20",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to('cuda')
    config = CompilationConfig.Default()
    # xformers and Triton are suggested for achieving best performance.
    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')
    config.enable_cuda_graph = True

    pipeline = compile(pipeline, config)
    for _ in range(2):
        pipeline(prompt="", num_inference_steps=14)

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=14,
    ).images[0]
