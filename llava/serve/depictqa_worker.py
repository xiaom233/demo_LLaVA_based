import string
from llava.model.depict.openlamm import LAMMPEFTModel
from model_worker import ModelWorker
import uuid
import threading
import time
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.constants import WORKER_HEART_BEAT_INTERVAL
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
import json
from threading import Thread
from transformers import TextIteratorStreamer
import random
import os
import argparse


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

QUALITY_COMPARE_SYS = "You are an AI visual assistant that can analyze images. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a quality comparison task, and your goal is to compare the quality between two images (Image A and Image B) from different aspects, such as, brightness, color, noise, artifacts, blur, and the texture damage. You are also given a Reference Image to assist your evaluation, which is a high-quality image with the same content as the images to be compared. You are encouraged to analyze the damage of specific contents. The quality analysis task involves understanding the user's input, generating an appropriate response about quality analysis, and maintaining a coherent conversation."

QUALITY_COMPARE_NOREASON_SYS = "You are an AI visual assistant that can analyze images. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a quality comparison task, and your goal is to compare the quality between two images (Image A and Image B). You are also given a Reference Image to assist your evaluation, which is a high-quality image with the same content as the images to be compared. The quality analysis task involves understanding the user's input, generating an appropriate response about quality analysis, and maintaining a coherent conversation."

QUALITY_SINGLE_SYS = "You are an AI visual assistant that can analyze images. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a quality analysis task, and your goal is to evaluate the quality of an image from different aspects, such as, brightness, color, noise, artifacts, blur, and texture damage. You are also given a Reference Image to assist your evaluation, which is a high-quality image with the same content as the image to be evaluated. You are encouraged to analyze the damage of specific contents. The quality analysis task involves understanding the user's input, generating an appropriate response about quality analysis, and maintaining a coherent conversation."

model_params = {"model": "lamm_peft",
                "encoder_pretrain": "clip",
                "vicuna_ckpt_path": r"C:\Users\Hoven_Li\Documents\GitHub\demo_LLaVA_based\model_zoo\vicuna_ckpt",
                "delta_ckpt_path": r"C:\Users\Hoven_Li\Documents\GitHub\demo_LLaVA_based\model_zoo\delta_cjpt",
                "max_tgt_len": 400,
                "lora_r": 16,
                "lora_alpha": 16,
                "ora_dropout": 0.1,
                "size_COCO": 224,
                "num_vision_token": 256,
                "vision_output_layer": -2,
                "conv_mode": "simple",
                "dataset-name": "test200_compare"
                }


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def get_sys_msg(task):
    if "quality" not in task:
        raise ValueError("This task is not supported yet")
    if "description" in task:
        return QUALITY_SINGLE_SYS
    elif "quality comparison" in task:
        return QUALITY_COMPARE_NOREASON_SYS
    elif "reasoning" in task:
        return QUALITY_COMPARE_SYS
    else:
        raise ValueError("This task is not supported yet")


class DepictQA(ModelWorker):
    def __init__(self, controller_addr, worker_addr, worker_id, no_register, model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        super().__init__(controller_addr, worker_addr, worker_id, no_register, model_path, model_base, model_name,
                         load_8bit, load_4bit, device)
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        # if model_name is None:
        #     model_paths = model_path.split("/")
        #     if model_paths[-1].startswith('checkpoint-'):
        #         self.model_name = model_paths[-2] + "_" + model_paths[-1]
        #     else:
        #         self.model_name = model_paths[-1]
        if model_name is None:
            self.model_name = "DepictQA"
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #     model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        self.model = LAMMPEFTModel(**model_params)
        delta_ckpt = torch.load(model_params["delta_ckpt_path"], map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
        self.model = self.model.eval().half().cuda()
        logger.info(f'[!] init the LLM over ...')

        self.is_multimodal = 'llava' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        task = params["task"]
        sys_msg = get_sys_msg(task)
        images = params.get("images", None)
        num_image_tokens = 0
        image_paths = []
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                for image in images:
                    image_path = os.path.join("images", str(random.sample(string.ascii_letters+string.digits, 8)))
                    image_paths.append(image_path)
                    image.save(image_path)
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}
        if task == "quality description":
            image_paths = image_paths[-2:].append("")
        elif task == "quality comparison" or "quality comparison and reasoning":
            image_paths = image_paths[-3:]
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.",
                              "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=model.generate, kwargs=dict(
            prompt=ori_prompt,
            images=image_paths[0],
            images_A=image_paths[1],
            images_B=image_paths[2],
            top_p=top_p,
            temperature=temperature,
            max_tgt_len=max_new_tokens,
            task_type=task,
            sys_msg=sys_msg,
            use_multi_tags=True,
            # from llava
            inputs=input_ids,
            do_sample=do_sample,
            # temperature=temperature,
            # top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"



