import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import faiss
import torch
import numpy as np
import open_clip
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE, ES_PASSWORD, IMAGE_WEIGHT, TEXT_WEIGHT
from elasticsearch import Elasticsearch
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

USERNAME = "vladimir"
PASSWORD = "vladimir1234"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False
)
USER_INTERACTIONS_INDEX = "user_interactions"

index = faiss.read_index(os.path.join(BASE_DIR, "creators.index"))
creators = [c.strip() for c in open(os.path.join(BASE_DIR, "creators.txt")).read().splitlines()]

img_embs = np.load(os.path.join(BASE_DIR, "image_embs.npy"))
txt_embs = np.load(os.path.join(BASE_DIR, "text_embs.npy"))

model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
model.eval().to(DEVICE).half()

print(f"✅ Loaded {len(creators)} creators")
print("✅ CLIP model ready")
print("✅ Elasticsearch connected")

def get_user_liked_creators(username, query_filter=None):
    try:
        if query_filter:
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"username": username}},
                            {"match": {"query_context": query_filter}}
                        ]
                    }
                },
                "size": 100,
                "_source": ["liked_creator"]
            }
        else:
            query_body = {
                "query": {"term": {"username": username}}, 
                "size": 100, 
                "_source": ["liked_creator"]
            }
        
        response = es.search(index=USER_INTERACTIONS_INDEX, body=query_body)
        return [hit["_source"]["liked_creator"] for hit in response["hits"]["hits"]]
    except:
        return []

def get_creator_embedding(creator_name):
    try:
        idx = creators.index(creator_name)
        combined = IMAGE_WEIGHT * img_embs[idx] + TEXT_WEIGHT * txt_embs[idx]
        combined /= np.linalg.norm(combined)
        return combined
    except:
        return None

def search_creators(query, username=None, top_k=20):
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokens = tokenizer([query]).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features[0].cpu().numpy().astype("float32")
    
    scores, idxs = index.search(np.expand_dims(query_vec, axis=0), top_k * 2)
    
    liked_creators = []
    if username:
        # Only get likes for THIS specific query
        liked_creators = get_user_liked_creators(username, query_filter=query)
    
    results = []
    for idx, score in zip(idxs[0], scores[0]):
        creator_name = creators[idx]
        personalized_score = float(score) * 0.7
        
        if liked_creators:
            creator_emb = get_creator_embedding(creator_name)
            if creator_emb is not None:
                similarity_scores = []
                for liked in liked_creators[:10]:
                    liked_emb = get_creator_embedding(liked)
                    if liked_emb is not None:
                        sim = np.dot(creator_emb, liked_emb)
                        similarity_scores.append(sim)
                
                if similarity_scores:
                    avg_similarity = np.mean(similarity_scores)
                    personalized_score += avg_similarity * 0.3
        
        results.append({'username': creator_name, 'score': personalized_score, 'original_score': float(score)})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('search'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('search'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    results = None
    query = ''
    username = session.get('username', 'anonymous')
    liked_count = len(get_user_liked_creators(username))
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            results = search_creators(query, username=username, top_k=20)
    
    return render_template('search.html', results=results, query=query, liked_count=liked_count)

@app.route('/like', methods=['POST'])
def like_creator():
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    data = request.get_json()
    creator_name = data.get('creator')
    query_context = data.get('query', '')
    username = session.get('username', 'anonymous')
    
    if not creator_name:
        return jsonify({'success': False, 'error': 'No creator specified'}), 400
    
    try:
        doc = {
            "username": username,
            "liked_creator": creator_name,
            "query_context": query_context,
            "timestamp": datetime.now().isoformat()
        }
        es.index(index=USER_INTERACTIONS_INDEX, document=doc)
        return jsonify({'success': True, 'message': f'Liked {creator_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
