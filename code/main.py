from utils import *
from caption import process_image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = input("GPUs: ")  # Only GPU 0 will be visible

# main.py
def run(folder_path, query, model, tokenizer, DEVICE, TORCH_TYPE):
    image_files = get_image_files(folder_path)
    images = load_images_to_ram(image_files)
    
    for file in tqdm(image_files, desc="Processing images"):
        try:
            process_image(file, images, query, model, tokenizer, DEVICE, TORCH_TYPE)
        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    params = setup()
    tokenizer = load_tokenizer(params)
    model = load_model(params)
    folder_path = "../data"
    query = """
    I have a frame from a video that needs detailed captioning. 
    Please provide an accurate and comprehensive description. 
    Focus on movements, actions, expressions, and any other distinguishing details, even if they are subtle. 
    Ensure that the description captures the unique aspects of the image.
    """
    run(folder_path, query, model, tokenizer, params['DEVICE'], params['TORCH_TYPE'])
