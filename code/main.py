import os
import json
import random
from tqdm import tqdm
from utils import *
from caption import process_image

os.environ["CUDA_VISIBLE_DEVICES"] = input("GPUs: ")

# Path to the master JSON file
MASTER_JSON_FILE = "../output/master_captions.json"

def load_completed_subfolders():
    if not os.path.exists(MASTER_JSON_FILE):
        with open(MASTER_JSON_FILE, "w") as f:
            json.dump({}, f)
        return set()
    else:
        with open(MASTER_JSON_FILE, "r") as f:
            completed_folders = json.load(f)
            return set(completed_folders.keys())

def load_video_text(subfolder):
    captions_file = os.path.join(subfolder, "captions.txt")
    if os.path.exists(captions_file):
        with open(captions_file, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

def run(folder_path, query, model, tokenizer, DEVICE, TORCH_TYPE):
    # Get subfolders and shuffle ONCE for the entire process
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    random.shuffle(subfolders)

    completed_subfolders = load_completed_subfolders()

    master_data = {}

    # Outer tqdm for subfolders, with initial value set for resuming
    outer_pbar = tqdm(initial=len(completed_subfolders), total=len(subfolders), desc="Processing subfolders")
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        if subfolder_name in completed_subfolders:
            outer_pbar.update()
            continue  # Move to the next subfolder, PBar already updated

        image_files = get_image_files(subfolder)
        images = load_images_to_ram(image_files)

        subfolder_data = {"video_text": load_video_text(subfolder)}  # Initialize video_text

        # Inner tqdm for image files
        inner_pbar = tqdm(image_files, desc=f"Processing {subfolder_name}")

        for file in image_files:
            try:
                caption_text = process_image(file, images, query, model, tokenizer, DEVICE, TORCH_TYPE)

                # Extract file name from path using os.path.split
                _, file_name = os.path.split(file)
                file_name = os.path.splitext(file_name)[0]  # Remove extension

                subfolder_data[file_name] = caption_text  # Use file_name as key
                inner_pbar.update()

            except Exception as e:
                print(f"Error processing {file}: {e}")

        inner_pbar.close()

        # Update the master JSON safely after each subfolder is processed
        with open(MASTER_JSON_FILE + ".tmp", "w") as f:
            try:
                with open(MASTER_JSON_FILE, "r") as original_file:
                    data = json.load(original_file)
            except FileNotFoundError:
                data = {}
            data[subfolder_name] = subfolder_data  # Use subfolder_name as key
            json.dump(data, f, indent=4)
        os.replace(MASTER_JSON_FILE + ".tmp", MASTER_JSON_FILE)

        master_data.clear()  # Clear for the next subfolder
        outer_pbar.update()

    outer_pbar.close()

if __name__ == "__main__":
    params = setup()
    tokenizer = load_tokenizer(params)
    model = load_model(params)
    folder_path = "../data/output"
    query = """
    I have a frame from a video that needs detailed captioning. 
    Please provide an accurate and comprehensive description. 
    Focus on movements, actions, expressions, and any other distinguishing details, even if they are subtle. 
    Ensure that the description captures the unique aspects of the image.
    """
    run(folder_path, query, model, tokenizer, params['DEVICE'], params['TORCH_TYPE'])