import torch

from .constants import VISION_TAGS


def get_p_before_embeds(tokenizer, llama_model, inputs, image_embeds, image_A_embeds, image_B_embeds):
    def get_embeds_from_text(text, batch_size):
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(image_embeds.device)
        embeds = llama_model.model.model.embed_tokens(tokens.input_ids).expand(batch_size, -1, -1)
        return embeds

    eov, sov = VISION_TAGS["eov"], VISION_TAGS["sov"]
    use_multi_tags = inputs["use_multi_tags"]
    sys_msg = inputs["sys_msg"]
    batch_size = image_embeds.shape[0]
    if inputs["task_type"] in ["quality_compare", "quality_compare_noreason"]:
        if use_multi_tags:
            eov_final = eov['<image_B>']
            p_before_texts = [
                sys_msg + f"\n\n### Reference Image: {sov['<image_reference>']}",
                f"{eov['<image_reference>']}\n\n### Image A: {sov['<image_A>']}",
                f"{eov['<image_A>']}\n\n### Image B: {sov['<image_B>']}",
            ]
        else:
            eov_final = eov['default']
            p_before_texts = [
                sys_msg + f"\n\n### Reference Image: {sov['default']}",
                f"{eov['default']}\n\n### Image A: {sov['default']}",
                f"{eov['default']}\n\n### Image B: {sov['default']}",
            ]
        p_before_embeds_list = [get_embeds_from_text(text, batch_size) for text in p_before_texts]
        p_before_embeds_list.insert(3, image_B_embeds)
        p_before_embeds_list.insert(2, image_A_embeds)
        p_before_embeds_list.insert(1, image_embeds)
    elif inputs["task_type"] == "quality_single":
        eov_final = eov['default']
        if use_multi_tags:
            p_before_texts = [
                sys_msg + f"\n\n### Reference Image: {sov['<image_reference>']}",
                f"{eov['<image_reference>']}\n\n### Image: {sov['default']}",
            ]
        else:
            p_before_texts = [
                sys_msg + f"\n\n### Reference Image: {sov['default']}",
                f"{eov['default']}\n\n### Image: {sov['default']}",
            ]
        p_before_embeds_list = [get_embeds_from_text(text, batch_size) for text in p_before_texts]
        p_before_embeds_list.insert(2, image_A_embeds)
        p_before_embeds_list.insert(1, image_embeds)
    else:
        raise NotImplementedError
    p_before_embeds = torch.cat(p_before_embeds_list, dim=1)
    return p_before_embeds, eov_final
