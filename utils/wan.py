# A ComfyScript conversion of a simple WAN workflow
# TODO: Add more configurability and RIFE/MMAudio/etc...

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
import sys, random

#pos_prompt = sys.argv[1] # 'overhead shot of a large crowd of people in times square, then the drone camera zooms out'


def run(pos_prompt, output_prefix='ComfyUI-WAN', width=512, height=512, frames=21, steps=20, cfg=6, seed=0, sampler='uni_pc', scheduler='simple',
    neg_prompt = 'bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards'
        ):
    if not seed: seed = random.randint(1, 0xFFFFFFFF)
    with Workflow():
        model = UNETLoader('wan2.1_t2v_14B_fp8_scaled.safetensors', 'default')
        model = ModelSamplingSD3(model, 8)
        
        # TODO: Enable/Disable this, specify proper values from a table
        model = TeaCache(model, 'wan2.1_t2v_14B', 0.4, 3)
        model = CompileModel(model, 'default', 'inductor', False, False)
        
        clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
        clip_text_encode_positive_prompt_conditioning = CLIPTextEncode(pos_prompt, clip)
        clip_text_encode_negative_prompt_conditioning = CLIPTextEncode(neg_prompt, clip)
        latent = EmptyHunyuanLatentVideo(width, height, frames, 1)
        latent = KSampler(model, seed, steps, cfg, sampler, scheduler, clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, latent, 1)
        vae = VAELoader('wan_2.1_vae.safetensors')
        image = VAEDecode(latent, vae)
        SaveImage(image, output_prefix)
