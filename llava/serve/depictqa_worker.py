import string
from llava.model.depict.openlamm import LAMMPEFTModel
from .model_worker import ModelWorker
import uuid
import threading
import time
import asyncio
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.constants import WORKER_HEART_BEAT_INTERVAL
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, REF_IMAGE_TOKEN, IMGA_IMAGE_TOKEN, IMGB_IMAGE_TOKEN
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
import json
from threading import Thread
from transformers import TextIteratorStreamer
import random
import os
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from functools import partial
import argparse
import uvicorn


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
                "vicuna_ckpt_path": r"/home/zyli/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5/",
                "delta_ckpt_path": r"/data/zyli/projects/demo_LLaVA_based/model_zoo/1017_7b_detail4.9k_cnn30k_mix44k_trad40k_coco49k224s_imgsystags_r16/pytorch_model_ep5.pt",
                "max_tgt_len": 400,
                "lora_target_modules": ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                "lora_r": 16,
                "lora_dropout": 0.1,
                "lora_alpha": 16,
                "ora_dropout": 0.1,
                "size_COCO": 224,
                "num_vision_token": 256,
                "vision_output_layer": -2,
                "conv_mode": "simple",
                "dataset-name": "test200_compare",
                "vision_feature_type": "local"
                }


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def get_sys_msg(task):
    if "quality" not in task:
        raise ValueError("This task is not supported yet")
    if "quality_single" in task:
        return QUALITY_SINGLE_SYS
    elif "quality_compare" in task:
        return QUALITY_COMPARE_NOREASON_SYS
    elif "quality_compare_noreason" in task:
        return QUALITY_COMPARE_SYS
    else:
        raise ValueError("This task is not supported yet")


class ModelThread(threading.Thread):
    """
    处理task相关的线程类
    """

    def __init__(self, func, kwargs):
        super(ModelThread, self).__init__()
        self.func = func  # 要执行的task类型
        self.kwargs = kwargs  # 要传入的参数
        self.result = None

    def run(self):
        # 线程类实例调用start()方法将执行run()方法,这里定义具体要做的异步任务
        print("start func {}".format(self.func.__name__))  # 打印task名字　用方法名.__name__
        self.result = self.func(self.kwargs)  # 将任务执行结果赋值给self.result变量

    def join(self, timeout=None):
        super(ModelThread, self).join()
        return self.result


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

        self.is_multimodal = True

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
        self.image_paths = []

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        task = params["task"]
        logger.info(f"Task: {task}")
        sys_msg = get_sys_msg(task)
        images = params.get("images", None)
        if images is None:
            logger.warning("No images received")
        else:
            logger.info("images received")
        num_image_tokens = 0
        image_path = "/home"
        ref_path = None
        a_path = None
        b_path = None
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                logger.info(f"images length is {len(images)}")

                images = [load_image_from_base64(image) for image in images]
                for image in images:
                    image_path = os.path.join("images", ''.join(random.sample(string.ascii_letters + string.digits, 8))+".png")
                    image.save(image_path)
                    logger.info(f"image path:")
                    logger.info(f"{image_path}")
                    self.image_paths.append(image_path)
            if len(images) == 1:
                ref_path = self.image_paths[0]
            elif len(images) >= 2:
                ref_path = self.image_paths[0]
                a_path = self.image_paths[1]
                b_path = self.image_paths[2]
            else:
                images = None
            image_args = {}
        else:
            images = None
            image_args = {}

        logger.info(f"Reference path: {ref_path}")
        logger.info(f"Image A path: {a_path}")
        logger.info(f"Image B path: {b_path}")

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = params.get('max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.",
                              "error_code": 0}).encode() + b"\0"
            return
        logger.info(f"prompt: {ori_prompt}")
        input = dict(
            prompt=ori_prompt,
            # images=image_paths[0],
            # images_A=image_paths[1],
            # images_B=image_paths[2],
            images=ref_path,
            images_A=a_path,
            images_B=b_path,
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
        )
        thread = ModelThread(func=model.generate, kwargs=input)
        thread.start()
        result = thread.join()

        generated_text = ori_prompt
        result_txt = result[0]
        print("Model output:")
        print(generated_text)

        for new_text in result_txt:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"



app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = DepictQA(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
