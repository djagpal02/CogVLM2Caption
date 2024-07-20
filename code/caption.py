from PIL import Image
import torch
from utils import get_image_files, load_images_to_ram, create_text_file_for_image
import io

def caption(image_data, query, model, tokenizer, DEVICE, TORCH_TYPE):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    query = "Human: " + query

    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=[],
        images=[image],
        template_version='chat'
    )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }

    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def process_image(file, images, query, model, tokenizer, DEVICE, TORCH_TYPE):
    image_data = images[file]
    return caption(image_data, query, model, tokenizer, DEVICE, TORCH_TYPE)