import requests
from PIL import Image
import gradio as gr
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(image):
    file_names = pd.read_csv('path.csv')
    annoy_to_file = file_names.values.tolist()
    response = requests.post("http://api-app:5000/reco", json={"image" :image.tolist()})
    paths = [element for sous_liste in [annoy_to_file[idx] for idx in response.json()] for element in sous_liste]
    logging.info('Le chemin : %s', paths)   
    images = []
    for path in paths:
        img = Image.open(path)
        images.append(img.convert("RGB"))  

    # Convert the 5 images to a single image
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    img_final = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        img_final.paste(img, (x_offset, 0))
        x_offset += img.width
    return img_final


def process_text(text):
    response = requests.post("http://api-app:5000/prompt_text", json={"description": text})
    data = response.json()
    titles = data[0]
    descriptions = data[1]
    # Output in the form {title : resume of the film}
    output = "\n\n".join([f"{title} : {description}" for title, description in zip(titles, descriptions)])
    return output


if __name__=='__main__':

    with gr.Blocks() as blocks:
            gr.Interface(fn=process_image, inputs="image", outputs="image", title="Film recommendation by poster similarity")
            gr.Interface(fn=process_text, inputs="text", outputs="text", title="Film recommendation by description similarity")
    blocks.launch(server_name = "0.0.0.0", server_port = 7860, share = True)  