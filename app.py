from flask import Flask, render_template, request, session, redirect, url_for, flash
from pymongo import MongoClient
import bcrypt
import math
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = "una-chiave-super-segreta"  # cambia con una chiave forte

# === DB e FAISS CONFIG ===
client = MongoClient("mongodb://172.17.224.1:27017")
db = client["arxiv_db"]
papers_collection = db["papers"]
categories_collection = db["categories"]
users_collection = db["users"]

INDEX_FILE = "faiss_papers_ivfflat.index"
IDS_FILE = "paper_ids.npy"
TOP_K = 5
model = SentenceTransformer("all-MiniLM-L6-v2")

# === FUNZIONI PER PAPER SIMILI ===

def load_index_and_ids():
    index = faiss.read_index(INDEX_FILE)
    ids = np.load(IDS_FILE, allow_pickle=True)
    return index, ids

def get_embedding_by_id(collection, paper_id):
    doc = collection.find_one({"id": paper_id}, {"embedding": 1, "abstract": 1})
    if not doc:
        return None
    if "embedding" in doc:
        return np.array(doc["embedding"], dtype=np.float32)
    elif "abstract" in doc and doc["abstract"]:
        return model.encode(doc["abstract"]).astype(np.float32)
    else:
        return None

def get_related_papers(paper_id):
    index, ids = load_index_and_ids()
    emb = get_embedding_by_id(papers_collection, paper_id)
    if emb is None:
        return [] 

    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    index.nprobe = 20

    distances, indices = index.search(emb, TOP_K + 1)
    related_ids = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        candidate_id = ids[idx]
        if candidate_id == paper_id:
            continue
        related_ids.append(candidate_id)
        if len(related_ids) >= TOP_K:
            break

    return list(papers_collection.find({"id": {"$in": related_ids}}))

# === ROTTE FLASK ===

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        next_page = request.form.get("next") or url_for("home")

        user = users_collection.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            session["admin"] = True
            return redirect(next_page)
        else:
            return render_template("login.html", error="Credenziali errate.", next=next_page)

    next_page = request.args.get("next") or request.referrer or url_for("home")
    return render_template("login.html", next=next_page)


@app.route("/logout")
def logout():
    next_page = request.referrer or url_for("home")
    session.pop("admin", None)
    return redirect(next_page)

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 100

    if not query:
        return render_template("results.html", results=[], query=query, total_pages=0, current_page=1, block_start=1, block_end=1)

    category_map = {
        cat["id"]: cat["title"] for cat in categories_collection.find()
    }

    mongo_query = { "title": { "$regex": query, "$options": "i" } }
    papers_all = list(papers_collection.find(mongo_query).limit(1000))
    total_results = len(papers_all)
    total_pages = math.ceil(total_results / per_page)

    block_start = ((page - 1) // 10) * 10 + 1
    block_end = min(block_start + 9, total_pages)

    papers = papers_all[(page - 1) * per_page : page * per_page]

    results = []
    for paper in papers:
        category_id = paper.get("categories")
        category_title = category_map.get(category_id, category_id)
        results.append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "update_date": paper.get("update_date"),
            "category_title": category_title,
            "abstract": paper.get("abstract", "").strip()[:],
            "doi": paper.get("doi")
        })

    return render_template(
        "results.html",
        results=results,
        query=query,
        total_pages=total_pages,
        current_page=page,
        block_start=block_start,
        block_end=block_end
    )

@app.route("/admin/delete/<paper_id>", methods=["POST"])
def delete_paper(paper_id):
    if not session.get("admin"):
        flash("Access denied.", "danger")
        return redirect(url_for("home"))

    result = papers_collection.delete_one({"id": paper_id})
    if result.deleted_count:
        flash("Paper successfully eliminated.", "success")
    else:
        flash("Error: paper not found.", "danger")

    return redirect(request.referrer or url_for("home"))

@app.route("/admin/edit/<paper_id>", methods=["GET", "POST"])
def edit_paper(paper_id):
    if not session.get("admin"):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("home"))

    paper = papers_collection.find_one({"id": paper_id})
    if not paper:
        flash("Paper not found.", "danger")
        return redirect(url_for("home"))

    if request.method == "POST":
        # Aggiorna i campi modificabili
        updated = {
            "title": request.form["title"],
            "authors": request.form["authors"],
            "doi": request.form["doi"],
            "categories": request.form["category"]
        }

        papers_collection.update_one({"id": paper_id}, {"$set": updated})
        flash("Paper updated successfully.", "success")

        # Recupera destinazione per redirect (pagina precedente)
        next_page = request.form.get("next") or url_for("search", q=paper["title"])
        return redirect(next_page)

    # Metodo GET → mostra il form precompilato
    categories = list(categories_collection.find())

    # Recupera pagina precedente per tornare indietro
    next_page = request.args.get("next") or request.referrer or url_for("home")
    return render_template("edit_paper.html", paper=paper, categories=categories, next=next_page)

@app.route("/related/<paper_id>")
def related(paper_id):
    related_papers = get_related_papers(paper_id)
    return render_template("partials/related_cards.html", related_papers=related_papers)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
