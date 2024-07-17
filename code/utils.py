import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model


def setup():
    MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    num_gpus = torch.cuda.device_count()
    quant = 0
    if 'int4' in MODEL_PATH:
        quant = 4
    elif num_gpus == 1 and torch.cuda.get_device_properties(0).total_memory < 48 * 1024 ** 3:
        quant = input("Quantization bits (4/8): ")
        quant = int(quant)

    return {"MODEL_PATH": MODEL_PATH, "DEVICE": DEVICE, "TORCH_TYPE": TORCH_TYPE, "num_gpus": num_gpus, "quant": quant}

def load_tokenizer(params):
    return AutoTokenizer.from_pretrained(params['MODEL_PATH'], trust_remote_code=True)

def load_model(params):
    if params['quant'] == 4:
        model = AutoModelForCausalLM.from_pretrained(
            params['MODEL_PATH'],
            torch_dtype=params['TORCH_TYPE'],
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True
        ).eval()
    elif params['quant'] == 8:
        model = AutoModelForCausalLM.from_pretrained(
            params['MODEL_PATH'],
            torch_dtype=params['TORCH_TYPE'],
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True
        ).eval()
    else:
        if params['num_gpus'] == 1:
            model = AutoModelForCausalLM.from_pretrained(
                params['MODEL_PATH'],
                torch_dtype=params['TORCH_TYPE'],
                trust_remote_code=True
            ).eval().to(params['DEVICE'])
        else:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    params['MODEL_PATH'],
                    torch_dtype=params['TORCH_TYPE'],
                    trust_remote_code=True,
                )

            max_memory_per_gpu = "16GiB"
            if params['num_gpus'] > 2:
                max_memory_per_gpu = f"{round(42 / params['num_gpus'])}GiB"

            device_map = infer_auto_device_map(
                model=model,
                max_memory={i: max_memory_per_gpu for i in range(params['num_gpus'])},
                no_split_module_classes=["CogVLMDecoderLayer"]
            )
            model = load_checkpoint_and_dispatch(model, params['MODEL_PATH'], device_map=device_map, dtype=params['TORCH_TYPE'], offload_folder="/tmp/offload")
            model = model.eval()

    return model

def get_image_files(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith((".jpg", ".png")):
                image_files.append(os.path.join(root, filename))
    return image_files

def load_images_to_ram(image_files):
    images = {}
    for file in image_files:
        with open(file, 'rb') as img_file:
            img_data = img_file.read()
            images[file] = img_data
    return images


def create_text_file_for_image(image_file, text):
    text_file_path = os.path.splitext(image_file)[0] + ".txt"
    with open(text_file_path, "w") as text_file:
        text_file.write(text)