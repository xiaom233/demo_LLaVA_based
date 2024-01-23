import io

import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn.utils import rnn

import conversations
from header import *
from transformers import StoppingCriteria, StoppingCriteriaList

from .CLIP import load as load_clip

from .constants import VISION_TAGS
from .modeling_llama import LlamaForCausalLM
from .quality_helper import get_p_before_embeds

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LAMMStoppingCriteria(StoppingCriteria):
    def __init__(self, stops, input_ids):
        """intialize stopping criteria

        :param list stops: list of stop tokens
        :param list input_ids: input ids
        """
        super().__init__()
        self.stops = [torch.tensor(stop).to('cuda') for stop in stops]
        self.stop_flag = [0] * input_ids.shape[0]

    def check_stop(self, input_ids):
        """check whether to stop generation

        :param list input_ids: input token ids
        :return bool: stop or not
        """
        for stop in self.stops:
            if torch.all((stop == input_ids[-len(stop):])).item():
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """call function of stop creteria

        :param torch.LongTensor output_ids: output token ids
        :return bool: stop or not
        """
        flag = 1
        # all batch stop -> return True
        for id, output_id in enumerate(output_ids):
            if self.stop_flag[id] == 1:
                continue
            if self.check_stop(output_id):
                self.stop_flag[id] = 1
            else:
                flag = 0
        if flag == 1:
            return True
        return False


def build_one_instance(tokenizer, conversation, task_type, use_multi_tags):
    """build one instance for training; text part

    :param class tokenizer: text tokenizer
    :param list conversation: list of conversation
    :raises Exception: Exception if wrong role included
    :return list: conversation text list, input token ids, target token ids
    """
    pos = VISION_TAGS["pos"]["default"]
    eov = VISION_TAGS["eov"]

    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn["from"]
        if i == 0:  # the first human turn
            assert role == "human"
            if use_multi_tags and "quality_compare" in task_type[0]:
                text = f"{eov['<image_B>']}\n\n### Human: " + turn["value"] + "\n### Assistant:"
            elif "quality" in task_type[0]:
                text = f"{eov['default']}\n\n### Human: " + turn["value"] + "\n### Assistant:"
            else:
                turn["value"] = (
                    turn["value"].replace(f"{pos}\n", "").replace(f"\n{pos}", "")
                )
                text = f"{eov['default']} " + turn["value"] + "\n### Assistant:"
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(
                one_input_id
            )  # do not perform loss regression on human prompt
        else:
            if role == "human":
                text = "Human: " + turn["value"] + "\n### Assistant:"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == "gpt":
                text = turn["value"] + "\n###"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception("Wrong Role!!!")
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len, task_type, use_multi_tags):
    """build one batch of instance for training

    :param class tokenizer: text tokenizer
    :param list batch_of_conversations: batch of conversations
    :param int max_tgt_len: max token length of after vision tokens
    :return list: input token ids, target token ids, attention mask
    """
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation, task_type, use_multi_tags)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    target_ids = rnn.pad_sequence(
        batch_target_ids, batch_first=True, padding_value=-100
    )
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def make_prompt_start(use_system=False, task_type="normal"):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    if "quality" in task_type[0]:
        PROMPT_START = "### "
    else:
        PROMPT_START = f'### Human: {VISION_TAGS["sov"]["default"]}'
    if use_system:
        if task_type == "normal":
            return f"{conversations.default_conversation.system}\n\n" + PROMPT_START
        else:
            return [
                f"{conversations.conversation_dict[task]}\n\n" + PROMPT_START
                for task in task_type
            ]
    else:
        return PROMPT_START


class LAMMPEFTModel(nn.Module):
    """LoRA for LAMM model"""

    def __init__(self, **args):
        super(LAMMPEFTModel, self).__init__()
        self.args = args
        encoder_ckpt_path = "~/.cache/clip/ViT-L-14.pt"
        vicuna_ckpt_path = args["vicuna_ckpt_path"]

        self.use_system = args["use_system"] if "use_system" in args else False
        self.use_multi_tags = args["use_multi_tags"] if "use_multi_tags" in args else False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing visual encoder from {encoder_ckpt_path} [{device}]...")

        # -1 for last embedding; -2 for transformer output
        self.vision_feature_type = args["vision_feature_type"]
        self.num_vision_token = args["num_vision_token"]

        size_COCO = args.get("size_COCO", 224)
        clip_encoder, self.visual_preprocess_quality, self.visual_preprocess = load_clip(
            "ViT-L/14", device=device, size_COCO=size_COCO
        )
        self.visual_encoder = clip_encoder.visual
        if self.vision_feature_type == "global":  # global feature from CLIP
            self.vision_hidden_size = 768
            self.num_vision_token = 1
            self.num_vision_token_quality = 1
        elif self.vision_feature_type == "local":  # patch features from CLIP
            self.vision_hidden_size = 1024
            self.num_vision_token = min(self.num_vision_token, 256)  # may cut partial tokens
            self.num_vision_token_quality = 25

        # freeze vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print("Visual encoder initialized.")

        print(f"Initializing language decoder from {vicuna_ckpt_path} ...")
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print("Language decoder initialized.")

        self.llama_proj = nn.Linear(
            self.vision_hidden_size, self.llama_model.config.hidden_size
        )
        print("LLaMa projection layer initialized.")

        self.max_tgt_len = args["max_tgt_len"]
        self.device = torch.cuda.current_device()

    def encode_image(self, image_paths, is_quality_analysis):
        """encode images to llama inputs

        :param tupe image_paths: (bsz, )
        :return tensor, tensor: input feature to llama, attention mask to llama
        """
        inputs = self.load_and_transform_image_data_clip(
            image_paths, self.device, is_quality_analysis
        )  # bsz x 3 x 224 x 224
        inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32
        inputs_llama = self.clip_encode_image(inputs, is_quality_analysis)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            self.device
        )  # bsz x 1/256
        return inputs_llama, atts_llama


    def clip_encode_image(self, inputs, is_quality_analysis):
        num_vision_token = self.num_vision_token_quality if is_quality_analysis else self.num_vision_token
        inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32

        if self.vision_feature_type == "global":
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)  # bsz x 768
            image_embeds = embeddings.to(self.llama_model.dtype)
            inputs_llama = self.llama_proj(image_embeds).unsqueeze(
                1
            )  # bsz x 1 x llama_size
        elif self.vision_feature_type == "local":
            with torch.no_grad():
                embeddings = self.visual_encoder.forward_patch_features(inputs)[
                    :, : num_vision_token
                ]  # bsz x num_vision_token x 1024
            image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(
                self.llama_model.dtype
            )  # bsz*num vision token x 1024
            inputs_llama = self.llama_proj(image_embeds).reshape(
                -1, num_vision_token, self.llama_model.config.hidden_size
            )  # bsz x num_vision_token x llama_size
        else:
            raise NotImplementedError(
                "{} not Implemented".format(self.vision_feature_type)
            )
        return inputs_llama

    def load_and_transform_image_data_clip(self, image_paths, device, is_quality_analysis):
        if image_paths is None:
            return None
        image_ouputs = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                image = Image.open(image_path)
            elif image_path.startswith("http://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                print("can not load image: ", image_path)
            if is_quality_analysis:
                image_output = self.visual_preprocess_quality(image).to(device)  # 3 x 224 x 224
            else:
                image_output = self.visual_preprocess(image).to(device)  # 3 x 224 x 224
            image_ouputs.append(image_output)
        return torch.stack(image_ouputs, dim=0)  # B x 3 x 224 x 224

    def prompt_wrap_quality(
        self, img_embeds_list, input_ids, target_ids, attention_mask, use_system, task_type
    ):
        """
        input_ids, target_ids, attention_mask: bsz x s2
        """
        def get_embeds_from_text(text, batch_size):
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
            embeds = self.llama_model.model.model.embed_tokens(tokens.input_ids).expand(batch_size, -1, -1)
            return embeds

        eov, sov = VISION_TAGS["eov"], VISION_TAGS["sov"]
        num_vision_token = self.num_vision_token_quality
        img_embeds, img_A_embeds, img_B_embeds = img_embeds_list
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2

        batch_size = img_embeds.shape[0]
        assert batch_size == 1, "prompt_wrap_quality only support batch_size == 1"

        msg_system = make_prompt_start(use_system=use_system, task_type=task_type)
        if "quality_compare" in task_type[0]:
            if self.use_multi_tags:
                p_before_texts = [
                    [msg_system[0] + "Reference Image: " + sov["<image_reference>"]],
                    [eov["<image_reference>"] + "\n\n### Image A: " + sov["<image_A>"]],
                    [eov["<image_A>"] + "\n\n### Image B: " + sov["<image_B>"]],
                ]
            else:
                p_before_texts = [
                    [msg_system[0] + "Reference Image: " + sov["default"]],
                    [eov["default"] + "\n\n### Image A: " + sov["default"]],
                    [eov["default"] + "\n\n### Image B: " + sov["default"]],
                ]
            p_before_embeds_list = [get_embeds_from_text(text, batch_size) for text in p_before_texts]
            p_before_embeds_list.insert(3, img_B_embeds)
            p_before_embeds_list.insert(2, img_A_embeds)
            p_before_embeds_list.insert(1, img_embeds)
            p_before_embeds = torch.cat(p_before_embeds_list, dim=1)
        elif "quality_single" in task_type[0]:
            if self.use_multi_tags:
                p_before_texts = [
                    [msg_system[0] + "Reference Image: " + sov["<image_reference>"]],
                    [eov["<image_reference>"] + "\n\n### Image: " + sov["default"]],
                ]
            else:
                p_before_texts = [
                    [msg_system[0] + "Reference Image: " + sov["default"]],
                    [eov["default"] + "\n\n### Image: " + sov["default"]],
                ]
            p_before_embeds_list = [get_embeds_from_text(text, batch_size) for text in p_before_texts]
            if task_type[0] == "quality_single_A":
                img_AB_embeds = img_A_embeds
            elif task_type[0] == "quality_single_B":
                img_AB_embeds = img_B_embeds
            p_before_embeds_list.insert(2, img_AB_embeds)
            p_before_embeds_list.insert(1, img_embeds)
            p_before_embeds = torch.cat(p_before_embeds_list, dim=1)
        else:
            raise ValueError

        p_before_attn_mask = torch.ones([*p_before_embeds.shape[:2]], dtype=torch.long).to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids) # bsz x s2 x embed_dim

        bos = (
            torch.ones([batch_size, 1], dtype=torch.long, device=self.device)
            * self.tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+N*NumImgToken+s2) x embed_dim

        # make target ids for prefix part
        empty_targets = (
            torch.ones(
                [batch_size, 1 + p_before_embeds.size()[1]],
                dtype=torch.long,
            )
            .to(self.device)
            .fill_(-100)  # 1 (bos) + s1
        )  # bsz x (1 + s1)
        targets = torch.cat(
            [empty_targets, target_ids], dim=1
        )  # bsz x (1+s1+N*NumImgToken+s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        # atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + num_vision_token], dtype=torch.long).to(self.device) # bsz x (1[bos] + s1 +num_image_tokens)
        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(
            self.device
        )  # bsz x 1
        attention_mask = torch.cat([atts_bos, p_before_attn_mask, attention_mask], dim=1)
        assert attention_mask.size() == targets.size()  # bsz x (1+s1+N*NumImgToken+s2)
        return inputs_embeds, targets, attention_mask

    def prompt_wrap(
        self, img_embeds, input_ids, target_ids, attention_mask, use_system, task_type
    ):
        """
        input_ids, target_ids, attention_mask: bsz x s2
        """
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2

        batch_size = img_embeds.shape[0]

        # return list of headers if multiple tasks
        p_before = make_prompt_start(use_system=use_system, task_type=task_type)
        if isinstance(p_before, list):
            p_before_tokens = [
                self.tokenizer(p, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .to(self.device)
                for p in p_before
            ]
            # TODO: test in batch
            p_before_token_ids = rnn.pad_sequence(
                p_before_tokens,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # bsz x s1
            p_before_attn_mask = p_before_token_ids.ne(
                self.tokenizer.pad_token_id
            )
        else:
            p_before_tokens = self.tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(
                self.device
            )  # [s1, s1...] list of batch size
            p_before_token_ids = p_before_tokens.input_ids.expand(
                batch_size, -1
            )  # bsz x s1
            p_before_attn_mask = p_before_tokens.attention_mask.expand(
                batch_size, -1
            )  # bsz x s1
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_token_ids
        )  # .expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(
            batch_size, -1, -1
        )  # bsz x s2 x embed_dim
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_token_ids.dtype,
                device=self.device,
            )
            * self.tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(
            bos
        )  # bsz x 1 x embed_dim
        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+NumToken+s2) x embed_dim

        # make target ids for prefix part
        empty_targets = (
            torch.ones(
                [batch_size, 1 + p_before_embeds.size()[1] + self.num_vision_token],
                dtype=torch.long,
            )
            .to(self.device)
            .fill_(-100)  # 1 (bos) + s1 + num_image_tokens (image vector)
        )  # bsz x (1 + s1 + 1)
        targets = torch.cat(
            [empty_targets, target_ids], dim=1
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        # atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + self.num_vision_token], dtype=torch.long).to(self.device) # bsz x (1[bos] + s1 +num_image_tokens)
        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(
            self.device
        )  # bsz x 1
        atts_img = torch.ones([batch_size, self.num_vision_token], dtype=torch.long).to(
            self.device
        )  # bsz x num_image_tokens
        attention_mask = torch.cat(
            [atts_bos, p_before_attn_mask, atts_img, attention_mask], dim=1
        )
        assert (
            attention_mask.size() == targets.size()
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        return inputs_embeds, targets, attention_mask

    def forward(self, inputs):
        task_type = inputs["task_type"]
        batch_size = len(task_type)

        # check quality_analysis
        is_quality_analysis = False
        if "quality" in task_type[0]:
            is_quality_analysis = True
            assert batch_size == 1, "quality tasks only support batch_size == 1"

        vision_paths = inputs["vision_paths"]
        vision_A_paths = inputs["vision_A_paths"]
        vision_B_paths = inputs["vision_B_paths"]
        vision_embeds, _ = self.encode_image(vision_paths, is_quality_analysis)
        if is_quality_analysis:
            vision_A_embeds, _ = self.encode_image(vision_A_paths, is_quality_analysis)
            vision_B_embeds, _ = self.encode_image(vision_B_paths, is_quality_analysis)

        output_texts = inputs["output_texts"]
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.tokenizer, output_texts, self.max_tgt_len, task_type, self.use_multi_tags
        )
        if is_quality_analysis:
            inputs_embeds, targets, attention_mask = self.prompt_wrap_quality(
                [vision_embeds, vision_A_embeds, vision_B_embeds],
                input_ids,
                target_ids,
                attention_mask,
                self.use_system,
                task_type,
            )
        else:
            inputs_embeds, targets, attention_mask = self.prompt_wrap(
                vision_embeds,
                input_ids,
                target_ids,
                attention_mask,
                self.use_system,
                task_type,
            )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(
            torch.long
        )  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def prepare_generation_embedding(self, inputs):
        """prepare for generation

        :param class inputs: model
        :return Dict: generation input
        """
        eov = VISION_TAGS["eov"]["default"]
        # TODO: add System header & image token size
        prompt_list = inputs["prompt"]  # questions from user
        feature_embeds, _ = self.encode_image(inputs["images"], is_quality_analysis=False)

        batch_size = feature_embeds.shape[0]
        p_before = make_prompt_start()  # no system header in test
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_tokens.input_ids
        ).expand(
            batch_size, -1, -1
        )  # bsz x s1 x embed_dim

        p_after_texts = [f"{eov} " + prompt + "\n### Assistant:" for prompt in prompt_list]
        p_after_tokens = self.tokenizer(
            p_after_texts, 
            padding="longest", return_length=True, # padding right
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_tokens.input_ids.dtype,
                device=self.device,
            )
            * self.tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(
            bos
        )  # bsz x 1 x embed_dim

        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, feature_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+NumVisionToken+s2) x embed_dim
        
        # p_after_embeds are on right, so the pads are right, 
        # we need to move all inputs_embeds to right,
        # to make the pads on left
        tokens_len = inputs_embeds.shape[1] - p_after_masks_len
        new_inputs_embeds = torch.zeros_like(inputs_embeds)
        inputs_embeds_masks = torch.ones(inputs_embeds.shape[:-1], 
                                         dtype=torch.int64, device=self.device)
        for idx in range(batch_size):
            # 1. If do_sample=True, mask pad tokens will cause some errors. 
            # 2. If do_sample=False, mask pad tokens will cause empty string: "". 
            # inputs_embeds_masks[idx, :-tokens_len[idx]] = 0
            new_inputs_embeds[idx, -tokens_len[idx]:, :] = inputs_embeds[idx, :tokens_len[idx], :]
            new_inputs_embeds[idx, :-tokens_len[idx], :] = inputs_embeds[idx, tokens_len[idx]:, :]

        return new_inputs_embeds, inputs_embeds_masks

    def prepare_generation_embedding_quality(self, inputs):
        prompt_list = inputs["prompt"]  # questions from user
        image_embeds, _ = self.encode_image(inputs["images"], is_quality_analysis=True)
        image_A_embeds, _ = self.encode_image(inputs["images_A"], is_quality_analysis=True)
        image_B_embeds, _ = self.encode_image(inputs["images_B"], is_quality_analysis=True)
        batch_size = image_embeds.shape[0]

        p_before_embeds, eov_final = get_p_before_embeds(
            self.tokenizer, self.llama_model, inputs, image_embeds, image_A_embeds, image_B_embeds
        )
        p_after_texts = [f"{eov_final}\n\n### Human: " + prompt + "\n### Assistant:" for prompt in prompt_list]
        p_after_tokens = self.tokenizer(
            p_after_texts, 
            padding="longest", return_length=True, # padding right
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

        bos = (torch.ones([batch_size, 1], dtype=torch.long, device=self.device)
            * self.tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim

        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1)

        # p_after_embeds are on right, so the pads are right, 
        # we need to move all inputs_embeds to right,
        # to make the pads on left
        tokens_len = inputs_embeds.shape[1] - p_after_masks_len
        new_inputs_embeds = torch.zeros_like(inputs_embeds)
        inputs_embeds_masks = torch.ones(inputs_embeds.shape[:-1], 
                                         dtype=torch.int64, device=self.device)
        for idx in range(batch_size):
            # 1. If do_sample=True, mask pad tokens will cause some errors. 
            # 2. If do_sample=False, mask pad tokens will cause empty string: "". 
            # inputs_embeds_masks[idx, :-tokens_len[idx]] = 0
            new_inputs_embeds[idx, -tokens_len[idx]:, :] = inputs_embeds[idx, :tokens_len[idx], :]
            new_inputs_embeds[idx, :-tokens_len[idx], :] = inputs_embeds[idx, tokens_len[idx]:, :]

        return new_inputs_embeds, inputs_embeds_masks

    def generate(self, inputs):
        if "quality" in inputs["task_type"]:
            is_quality_analysis = True
            input_embeds, input_masks = self.prepare_generation_embedding_quality(inputs)
        else:
            is_quality_analysis = False
            input_embeds, input_masks = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList(
            [LAMMStoppingCriteria([[2277, 29937], [835]], input_embeds)]
        )
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            max_new_tokens=inputs["max_tgt_len"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            do_sample=False if is_quality_analysis else True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text
