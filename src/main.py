import openai
import os
import torch
import clip
from PIL import Image
import numpy as np
from config import OPENAI_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
import mplfinance as mpf

openai.api_key = OPENAI_API_KEY

# Load CLIP model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Read files
root_dir = os.path.dirname(os.path.abspath(__file__))
training_images_dir = os.path.join(root_dir, "..", "data/training_images")
training_images_file_paths = [os.path.join(training_images_dir, f) for f in os.listdir(training_images_dir) if os.path.isfile(os.path.join(training_images_dir, f))]

sample_images_dir = os.path.join(root_dir, "..", "data/sample_images")
sample_images_file_paths = [os.path.join(sample_images_dir, f) for f in os.listdir(sample_images_dir) if os.path.isfile(os.path.join(sample_images_dir, f))]

# Generates image embeddings
def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image)

    return embedding.cpu().numpy().flatten()

# Finds the closest matching chart based on cosine similarity
def get_closest_match(sample_image_path, training_embeddings):
    sample_embedding = get_image_embedding(sample_image_path)

    # Compute cosine similarity
    similarities = cosine_similarity([sample_embedding], training_embeddings)

    # Get index of the most similar chart
    closest_index = np.argmax(similarities)

    return training_images_file_paths[closest_index], similarities[0][closest_index]

def interpret_sample_file(file_name, training_embeddings):
    sample_image_path = os.path.join(sample_images_dir, file_name)
    closest_match, similarity_score = get_closest_match(sample_image_path, training_embeddings)
    closest_match_filename = os.path.basename(closest_match)
    price_movement_prediction = 'down' if closest_match_filename.startswith('d') else 'up'
    closest_match_chart_type = os.path.splitext(closest_match_filename)[0][2:]
    sample_image_filename = os.path.basename(sample_image_path)
    formatted_percentage_str = f"{similarity_score*100:.2f}%"
    print(f"\nSample file\t: {sample_image_filename}\nClosest match\t: {closest_match_filename}\nSimilarity score: {formatted_percentage_str}\n\033[1m{closest_match_chart_type} indicates price may go {price_movement_prediction}\033[0m\n")

def delete_all_sample_images():
    for file_name in os.listdir(sample_images_dir):
        file_path = os.path.join(sample_images_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def print_line():
    print("--------------------------------------------------------")

def main():
    # Generating embeddings for training images
    print("Program started.")
    training_embeddings = [get_image_embedding(img) for img in training_images_file_paths]
    training_embeddings = np.array(training_embeddings)
    print("Training image embeddings generated.\n")
    while True:
        user_input = input("Type 'delete' to delete all sample images\nType 'exit' to exit the program\nType a stock ticker to get candle chart interpretation\n: ")
        if user_input.lower() == "exit":
            print("Program ended.")
            break
        elif user_input.lower() == "delete":
            delete_all_sample_images()
            print("All sample files deleted.")
        ticker = user_input.upper()
        print_line()
        print("\nGetting candle stick chart for " + ticker)
        stock = yf.Ticker(ticker)
        df = stock.history(period="30d")
        new_file_name = ticker +".png"
        mpf.plot(df, type="candle", style="charles", volume=True, savefig="data/sample_images/" + new_file_name)
        print("Candle stick chart " + new_file_name + " saved.\nInterpretting new chart...")
        interpret_sample_file(new_file_name, training_embeddings)
        print_line()
main()