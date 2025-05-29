import os
import sys
import subprocess
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
import argparse
from PIL import Image
import torch
from inference_solver import FlexARInferenceSolver
sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])
from xllmx.util.misc import random_seed
import time
from jacobi_utils_static import renew_pipeline_sampler

def download_model_if_missing(model_ckpt_path):
    if not os.path.exists(model_ckpt_path):
        print(f"Model checkpoint not found at {model_ckpt_path}, downloading...")
        os.makedirs(os.path.dirname(model_ckpt_path), exist_ok=True)
        subprocess.run([
            "wget",
            "-O", model_ckpt_path,
            "https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_270M.ckpt"
        ], check=True)
    else:
        print("✅ Model checkpoint already exists.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--task", type=str, default='t2i')
    parser.add_argument("--speculative_jacobi", default=False, action='store_true')
    parser.add_argument("--quant", default=False, action='store_true')

    args = parser.parse_args()

    print("args:\n", args)

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height
    device = torch.device("cuda")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if os.path.isdir(args.model_path):
        model_ckpt = os.path.join(args.model_path, "movqgan/270M/movqgan_270M.ckpt")
    else:
        model_ckpt = args.model_path

    print(f"Using model checkpoint path: {model_ckpt}")

    download_model_if_missing(model_ckpt)

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        quant=args.quant,
        sjd=args.speculative_jacobi,
    )
    print("checkpoint load finished")
    
    model_ckpt = os.path.join(args.model_path, "lumina_mgpt/movqgan/270M/movqgan_270M.ckpt")
    download_model_if_missing(model_ckpt)

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        quant=args.quant,
        sjd=args.speculative_jacobi,
    )
    print("checkpiont load finished")

    if args.speculative_jacobi:
        print(inference_solver.__class__)
        print("Use Speculative Jacobi Decoding to accelerate!")
        max_num_new_tokens = 16
        multi_token_init_scheme = 'random' # 'repeat_horizon'
        inference_solver = renew_pipeline_sampler(
            inference_solver,
            jacobi_loop_interval_l = 3,
            jacobi_loop_interval_r = (h // 8)**2 + h // 8 - 10,
            max_num_new_tokens = max_num_new_tokens,
            guidance_scale = cfg,
            seed = None,
            multi_token_init_scheme = multi_token_init_scheme,
            do_cfg=True,
            image_top_k=top_k, 
            text_top_k=10,
            prefix_token_sampler_scheme='speculative_jacobi',
            is_compile=args.quant
        )
        
    input_img = Image.open("input_images/img_046.jpeg").convert("RGB")
    prompt = "Make a black and white coloring book outline from the given photo"

    with torch.no_grad():
        for repeat_idx in range(n):
            random_seed(repeat_idx)
            generated = inference_solver.generate(
                images=[input_img],
                qas=[[prompt, None]],
                max_gen_len=10240,
                temperature=t,
                logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
            )
            generated[1][0].save(os.path.join(args.save_path, f"result_{repeat_idx}.png"))        
