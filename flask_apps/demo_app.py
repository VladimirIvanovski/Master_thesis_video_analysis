import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import faiss
import torch
import numpy as np
import open_clip
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, DEVICE, ES_PASSWORD, IMAGE_WEIGHT, TEXT_WEIGHT, RESULTS_DIR
from elasticsearch import Elasticsearch
from datetime import datetime
import glob

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

USERNAME = "vladimir"
PASSWORD = "vladimir123"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ABS_DIR = os.path.join(BASE_DIR, RESULTS_DIR)

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

def get_creator_frames(creator_name, max_frames=3):
    creator_path = os.path.join(RESULTS_ABS_DIR, creator_name)
    if not os.path.exists(creator_path):
        return []
    
    frames = glob.glob(os.path.join(creator_path, "**/frames/*.png"), recursive=True)
    frames = sorted(frames)[:max_frames]
    return [os.path.relpath(f, RESULTS_ABS_DIR).replace('\\', '/') for f in frames]

def normalize_query(query):
    return " ".join((query or "").strip().lower().split())

def get_user_feedback(username, query_filter=None):
    try:
        response = es.search(
            index=USER_INTERACTIONS_INDEX,
            body={
                "query": {"term": {"username": username}},
                "size": 500,
                "_source": ["liked_creator", "query_context", "label"]
            }
        )
        hits = [hit["_source"] for hit in response["hits"]["hits"]]
        if query_filter:
            q = normalize_query(query_filter)
            hits = [h for h in hits if normalize_query(h.get("query_context", "")) == q]

        good = []
        bad = []
        for h in hits:
            creator = h.get("liked_creator")
            if not creator:
                continue
            label = (h.get("label") or "good").lower()
            if label == "bad":
                bad.append((creator, h.get("query_context", "")))
            else:
                good.append((creator, h.get("query_context", "")))

        return {"good": good, "bad": bad}
    except:
        return {"good": [], "bad": []}

def get_creator_embedding(creator_name):
    try:
        idx = creators.index(creator_name)
        combined = IMAGE_WEIGHT * img_embs[idx] + TEXT_WEIGHT * txt_embs[idx]
        combined /= np.linalg.norm(combined)
        return combined
    except:
        return None

def search_creators_unpersonalized(query, top_k=10):
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokens = tokenizer([query]).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features[0].cpu().numpy().astype("float32")
    
    scores, idxs = index.search(np.expand_dims(query_vec, axis=0), top_k)
    
    results = []
    for idx, score in zip(idxs[0], scores[0]):
        creator_name = creators[idx]
        results.append({
            'username': creator_name,
            'score': float(score),
            'frames': get_creator_frames(creator_name)
        })
    
    return results

def search_creators_personalized(query, username, top_k=10):
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokens = tokenizer([query]).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features[0].cpu().numpy().astype("float32")
    
    scores, idxs = index.search(np.expand_dims(query_vec, axis=0), top_k * 3)
    
    # Only use feedback for THIS exact query context
    feedback = get_user_feedback(username, query_filter=query)
    good_creators = [x[0] for x in feedback["good"]]
    bad_creators = [x[0] for x in feedback["bad"]]
    
    results = []
    for idx, score in zip(idxs[0], scores[0]):
        creator_name = creators[idx]
        
        personalized_score = float(score) * 0.7
        similarity_boost = 0.0
        similarity_penalty = 0.0
        
        if good_creators:
            creator_emb = get_creator_embedding(creator_name)
            if creator_emb is not None:
                similarity_scores = []
                for liked in good_creators[:10]:
                    liked_emb = get_creator_embedding(liked)
                    if liked_emb is not None:
                        sim = np.dot(creator_emb, liked_emb)
                        similarity_scores.append(sim)
                
                if similarity_scores:
                    similarity_boost = np.mean(similarity_scores) * 0.25
                    personalized_score += similarity_boost

        if bad_creators:
            creator_emb = get_creator_embedding(creator_name)
            if creator_emb is not None:
                similarity_scores = []
                for disliked in bad_creators[:10]:
                    disliked_emb = get_creator_embedding(disliked)
                    if disliked_emb is not None:
                        sim = np.dot(creator_emb, disliked_emb)
                        similarity_scores.append(sim)
                if similarity_scores:
                    similarity_penalty = np.mean(similarity_scores) * 0.25
                    personalized_score -= similarity_penalty
        
        results.append({
            'username': creator_name,
            'score': personalized_score,
            'original_score': float(score),
            'similarity_boost': similarity_boost,
            'similarity_penalty': similarity_penalty,
            'frames': get_creator_frames(creator_name),
            'is_boosted': similarity_boost > 0.05,
            'is_penalized': similarity_penalty > 0.05
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('demo'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('demo'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    username = session.get('username', 'anonymous')
    all_feedback = get_user_feedback(username)  # All feedback for display
    liked_data = all_feedback["good"]
    disliked_data = all_feedback["bad"]
    liked_count = len(liked_data)
    disliked_count = len(disliked_data)
    
    unpersonalized_results = None
    personalized_results = None
    query = ''
    stats = None
    feedback_map = {}
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            query_feedback = get_user_feedback(username, query_filter=query)
            for creator, _ in query_feedback["good"]:
                feedback_map[creator] = "good"
            for creator, _ in query_feedback["bad"]:
                feedback_map[creator] = "bad"

            unpersonalized_results = search_creators_unpersonalized(query, top_k=10)
            personalized_results = search_creators_personalized(query, username, top_k=10)
            
            unpers_usernames = {r['username'] for r in unpersonalized_results}
            pers_usernames = {r['username'] for r in personalized_results}
            
            new_in_personalized = pers_usernames - unpers_usernames
            removed_from_personalized = unpers_usernames - pers_usernames
            
            boosted_count = sum(1 for r in personalized_results if r.get('is_boosted', False))
            penalized_count = sum(1 for r in personalized_results if r.get('is_penalized', False))
            
            stats = {
                'new_creators': len(new_in_personalized),
                'removed_creators': len(removed_from_personalized),
                'boosted_creators': boosted_count,
                'penalized_creators': penalized_count,
                'avg_adjustment': np.mean([
                    r.get('similarity_boost', 0) - r.get('similarity_penalty', 0)
                    for r in personalized_results
                ])
            }
    
    return render_template('demo.html', 
                          unpersonalized_results=unpersonalized_results,
                          personalized_results=personalized_results,
                          query=query, 
                          liked_count=liked_count,
                          disliked_count=disliked_count,
                          liked_data=liked_data,
                          disliked_data=disliked_data,
                          stats=stats,
                          feedback_map=feedback_map)

@app.route('/feedback', methods=['POST'])
def feedback_creator():
    if 'logged_in' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    data = request.get_json()
    creator_name = data.get('creator')
    query_context = data.get('query', '')
    label = (data.get('label') or 'good').lower()
    username = session.get('username', 'anonymous')
    
    if not creator_name or label not in {"good", "bad"}:
        return jsonify({'success': False, 'error': 'No creator specified'}), 400
    
    try:
        doc = {
            "username": username,
            "liked_creator": creator_name,
            "query_context": query_context,
            "label": label,
            "timestamp": datetime.now().isoformat()
        }
        es.index(index=USER_INTERACTIONS_INDEX, document=doc)
        return jsonify({'success': True, 'message': f'Saved {label} feedback for {creator_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    file_path = os.path.join(RESULTS_ABS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return '', 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
