from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from script_model.image_inference.mobileNet_model import MobileNet, process
import torch
from PIL import Image
import numpy as np
import logging
from script_model.nlp_inference.reco_nlp import preprocess, recommend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Load the Annoy database
annoy_db = AnnoyIndex(576, metric='angular')  
annoy_db.load('annoy_db_img.ann')  


@app.route('/') 
def index():    
    return 'Hello world!'

# This route is used to get recommendations from poster
@app.route('/reco', methods=['POST']) 
def reco():
    vector = np.array(request.json['image'],dtype=np.uint8) # Get the vector from the 
    model = MobileNet()
    model.eval()
    vector=Image.fromarray(vector).convert('RGB')
    image = process(1, vector)
    with torch.no_grad():
        vector = model(image.unsqueeze(0))
    closest_indices = annoy_db.get_nns_by_vector(vector[0].numpy(), 5) # Get the 5 closest elements indices  
    logging.info('La variable est : %s', closest_indices)
    return jsonify(closest_indices) 

# This route is used to get an user description of a film and return the 5 closest film
@app.route('/prompt_text', methods=['POST'])
def prompt():
    model, data = preprocess()
    request_usr = request.json
    query = request_usr.get('description')
    titles, descriptions = recommend(model, data, query, k = 5)
    return jsonify(titles, descriptions)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Run the server on port 5000 and make it accessible externally
