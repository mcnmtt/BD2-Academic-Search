import os
from flask import Flask, render_template, request, session, redirect, url_for, flash
from pymongo import MongoClient
import bcrypt
import math
import hnswlib
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

from datetime import datetime

app = Flask(__name__)
app.secret_key = 'b4b91b05e5e8491eb7a90cf536ad2dd927e48cc61dca5212ad9e3d03ec223f2d'

# === CONFIGURAZIONE GRAPH ===

@app.route("/graph/<paper_id>")
def graph(paper_id):
    return render_template("graph.html", paper_id=paper_id)

@app.route("/api/similarity_graph/<paper_id>")
def similarity_graph(paper_id):
    index, paper_ids = load_index_and_ids()
    paper_ids = list(paper_ids)

    if paper_id not in paper_ids:
        return {"error": "Paper ID not found in index"}, 404

    root_paper = papers_collection.find_one({"id": paper_id})
    if not root_paper or "embedding" not in root_paper:
        return {"error": "Paper not found or missing embedding"}, 404

    root_embedding = np.array(root_paper["embedding"], dtype=np.float32).reshape(1, -1)
    k = 50

    labels, distances = index.knn_query(root_embedding, k=k)

    nodes = []
    edges = []
    added_ids = set()

    # Nodo root
    nodes.append({
        "id": paper_id,
        "label": root_paper["title"][:60],
        "category": root_paper["categories"],
        "color": "red"
    })
    added_ids.add(paper_id)

    level1 = []
    for idx, dist in zip(labels[0], distances[0]):
        similar_id = str(paper_ids[idx])
        if similar_id == paper_id:
            continue
        similarity = 1 - dist
        if similarity >= 0.40:
            paper = papers_collection.find_one({"id": similar_id})
            if not paper or "embedding" not in paper:
                continue
            nodes.append({
                "id": similar_id,
                "label": paper["title"][:60],
                "category": paper["categories"],
                "color": "lightblue"
            })
            edges.append({
                "from": paper_id,
                "to": similar_id,
                "label": f"{similarity:.2f}"
            })
            added_ids.add(similar_id)
            level1.append((similar_id, np.array(paper["embedding"], dtype=np.float32).reshape(1, -1)))

    # Livello 2
    for leaf_id, leaf_emb in level1:
        labels2, distances2 = index.knn_query(leaf_emb, k=30)
        for idx2, dist2 in zip(labels2[0], distances2[0]):
            neighbor_id = str(paper_ids[idx2])
            if neighbor_id in added_ids or neighbor_id == paper_id or neighbor_id == leaf_id:
                continue
            similarity2 = 1 - dist2
            if similarity2 >= 0.50:
                paper = papers_collection.find_one({"id": neighbor_id})
                if not paper or "embedding" not in paper:
                    continue
                nodes.append({
                    "id": neighbor_id,
                    "label": paper["title"][:60],
                    "category": paper["categories"],
                    "color": "lightgreen"
                })
                edges.append({
                    "from": leaf_id,
                    "to": neighbor_id,
                    "label": f"{similarity2:.2f}"
                })
                added_ids.add(neighbor_id)

    return {"nodes": nodes, "edges": edges}


# === CONFIGURAZIONE DB E HNSW ===
client = MongoClient("mongodb://localhost:27017")
db = client["arvix_db"]
papers_collection = db["papers"]
categories_collection = db["categories"]
users_collection = db["users"]

INDEX_FILE = os.path.join("hnsw_index", "hnsw_papers.index")
IDS_FILE = os.path.join("hnsw_index", "paper_ids.npy")
TOP_K = 5
DIM = 384
model = SentenceTransformer("all-MiniLM-L6-v2")


def parse_authors(authors_str: str) -> list:
    """
    Converte la stringa di autori (uno per riga) in lista di stringhe.
    Esempio:
      "C. Balázs
       E. Berger
       P. M. Nadolsky"
    → ["C. Balázs", "E. Berger", "P. M. Nadolsky"]
    """
    if not authors_str:
        return []
    return [riga.strip() for riga in authors_str.splitlines() if riga.strip()]


# === FUNZIONI PER SIMILARITÀ ===

def load_index_and_ids():
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.load_index(INDEX_FILE)
    ids = np.load(IDS_FILE, allow_pickle=True)
    return index, [str(i) for i in ids]


def save_index_and_ids(index, ids):
    index.save_index(INDEX_FILE)
    np.save(IDS_FILE, np.array(ids, dtype=object))


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
    labels, distances = index.knn_query(emb, k=TOP_K + 1)

    related_results = []
    for idx, dist in zip(labels[0], distances[0]):
        candidate_id = ids[idx]
        if candidate_id == paper_id:
            continue
        paper = papers_collection.find_one({"id": candidate_id})
        if paper:
            similarity = round(1 - dist, 3)
            paper["similarity"] = similarity
            related_results.append(paper)
        if len(related_results) >= TOP_K:
            break

    # Ordina per similarità decrescente
    related_results.sort(key=lambda x: x["similarity"], reverse=True)
    return related_results


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
    per_page = 50

    if not query:
        return render_template("results.html", results=[], query=query, total_pages=0, current_page=1, block_start=1,
                               block_end=1)

    category_map = {cat["id"]: cat["title"] for cat in categories_collection.find()}

    mongo_query = {
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"authors": {"$regex": query, "$options": "i"}},
            {"abstract": {"$regex": query, "$options": "i"}},
            {"categories": {"$regex": query, "$options": "i"}}
        ]
    }

    papers_all = list(papers_collection.find(mongo_query).limit(500))
    total_results = len(papers_all)
    total_pages = math.ceil(total_results / per_page)

    block_start = ((page - 1) // 10) * 10 + 1
    block_end = min(block_start + 9, total_pages)

    papers = papers_all[(page - 1) * per_page: page * per_page]

    results = []
    for paper in papers:
        category_id = paper.get("categories")
        category_title = category_map.get(category_id, category_id)
        results.append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "authors": paper.get("authors", "").strip(),
            "update_date": paper.get("update_date"),
            "category_title": category_title,
            "abstract": paper.get("abstract", "").strip(),
            "doi": paper.get("doi")
        })

    return render_template("results.html", results=results, query=query, total_pages=total_pages,
                           current_page=page, block_start=block_start, block_end=block_end)


@app.route("/admin/delete/<paper_id>", methods=["POST"])
def delete_paper(paper_id):
    if not session.get("admin"):
        flash("Access denied.", "danger")
        return redirect(url_for("home"))

    # Rimuovi dal DB
    result = papers_collection.delete_one({"id": paper_id})

    if result.deleted_count:
        try:
            # === Rimuovi dall'indice HNSW ===
            index = hnswlib.Index(space="cosine", dim=DIM)
            index.load_index(INDEX_FILE)
            ids = list(np.load(IDS_FILE, allow_pickle=True))

            if paper_id in ids:
                hnsw_internal_id = ids.index(paper_id)
                index.mark_deleted(hnsw_internal_id)

                # Rimuovi ID da lista e salva aggiornamenti
                ids.pop(hnsw_internal_id)
                np.save(IDS_FILE, np.array(ids))
                index.save_index(INDEX_FILE)

                flash("Paper successfully eliminated and removed from index.", "success")
            else:
                flash("Paper deleted from DB, but not found in index.", "warning")

        except Exception as e:
            flash(f"Paper deleted from DB, but error removing from index: {str(e)}", "warning")
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
        updated = {
            "title": request.form["title"],
            "authors": request.form["authors"],
            "doi": request.form["doi"],
            "categories": request.form["category"]
        }
        papers_collection.update_one({"id": paper_id}, {"$set": updated})
        flash("Paper updated successfully.", "success")
        return redirect(request.form.get("next") or url_for("search", q=paper["title"]))

    categories = list(categories_collection.find())
    return render_template("edit_paper.html", paper=paper, categories=categories,
                           next=request.args.get("next") or request.referrer or url_for("home"))


@app.route("/admin/add", methods=["GET", "POST"])
def add_paper():
    if not session.get("admin"):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("home"))

    if request.method == "POST":
        paper_id = request.form.get("id", "").strip()
        title = request.form.get("title", "").strip()
        category_id = request.form.get("category", "").strip()
        authors_raw = request.form.get("authors", "").strip()
        abstract = request.form.get("abstract", "").strip()
        update_date_str = request.form.get("update_date", "").strip()

        submitter = request.form.get("submitter", "").strip()
        comments = request.form.get("comments", "").strip()
        journal_ref = request.form.get("journal_ref", "").strip()
        doi = request.form.get("doi", "").strip()
        report_no = request.form.get("report_no", "").strip()

        # Validazioni
        missing = []
        if not paper_id:
            missing.append("ID del Paper")
        if not title:
            missing.append("Titolo")
        if not category_id:
            missing.append("Categoria")
        if not abstract:
            missing.append("Abstract")
        if missing:
            flash(f"Errore: mancano i campi obbligatori: {', '.join(missing)}.", "danger")
            categories = list(categories_collection.find())
            return render_template("add_paper.html", categories=categories, form_data=request.form)

        if papers_collection.find_one({"id": paper_id}):
            flash(f"Errore: esiste già un paper con id = {paper_id}.", "danger")
            categories = list(categories_collection.find())
            return render_template("add_paper.html", categories=categories, form_data=request.form)

        if not categories_collection.find_one({"id": category_id}):
            flash(f"Errore: la categoria '{category_id}' non esiste.", "danger")
            categories = list(categories_collection.find())
            return render_template("add_paper.html", categories=categories, form_data=request.form)

        try:
            if update_date_str:
                update_date = datetime.strptime(update_date_str, "%Y-%m-%d").date().isoformat()
            else:
                raise ValueError
        except ValueError:
            update_date = datetime.utcnow().date().isoformat()

        authors_list = parse_authors(authors_raw)

        # Calcolo embedding
        try:
            embedding_np = model.encode(abstract).astype(np.float32)
            embedding_list = embedding_np.tolist()
        except Exception as e:
            flash(f"Errore nel calcolo embedding: {str(e)}", "danger")
            categories = list(categories_collection.find())
            return render_template("add_paper.html", categories=categories, form_data=request.form)

        # Documento finale
        new_doc = {
            "id": paper_id,
            "submitter": submitter or None,
            "authors": authors_raw or None,
            "authors_parsed": authors_list,
            "title": title,
            "comments": comments or None,
            "journal-ref": journal_ref or None,
            "doi": doi or None,
            "report-no": report_no or None,
            "categories": category_id,
            "license": None,
            "abstract": abstract,
            "update_date": update_date,
            "versions": [],
            "embedding": embedding_list
        }

        # Inserimento nel DB
        try:
            papers_collection.insert_one(new_doc)
        except Exception as e:
            flash(f"Errore durante inserimento in DB: {str(e)}", "danger")
            categories = list(categories_collection.find())
            return render_template("add_paper.html", categories=categories, form_data=request.form)

        # Inserimento nell'indice HNSW
        try:
            index, ids = load_index_and_ids()
            new_index = len(ids)
            index.add_items(np.array([embedding_np]), np.array([new_index]))
            ids = np.append(ids, paper_id)
            save_index_and_ids(index, ids)
        except Exception as e:
            flash(f"Paper inserito in DB ma errore indicizzazione HNSW: {str(e)}", "warning")
            return redirect(url_for("home"))

        flash(f"Paper '{paper_id}' inserito correttamente e indicizzato.", "success")
        return redirect(url_for("home"))

    # Se GET → mostra il form
    categories = list(categories_collection.find())
    return render_template("add_paper.html", categories=categories, form_data={})


@app.route("/related/<paper_id>")
def related(paper_id):
    related_papers = get_related_papers(paper_id)
    return render_template("partials/related_cards.html", related_papers=related_papers)


@app.route("/asn/<path:author>")
def asn_graph(author):
    return render_template("asn.html", author=author)


@app.route("/api/asn/<author>")
def authors_similarity_network(author):
    def extract_authors(authors_str):
        if not authors_str:
            return []
        authors_str = authors_str.replace(" and ", ",")
        return [a.strip() for a in authors_str.split(",") if a.strip()]

    papers = list(papers_collection.find({"authors": {"$regex": author, "$options": "i"}}))

    coauthors_map = defaultdict(Counter)

    for paper in papers:
        authors = extract_authors(paper.get("authors", ""))
        for a1 in authors:
            for a2 in authors:
                if a1 != a2:
                    coauthors_map[a1][a2] += 1

    nodes = []
    edges = []
    added = set()

    for author_name in coauthors_map:
        nodes.append({
            "id": author_name,
            "label": author_name,
            "color": "red" if author_name == author else "lightblue"
        })
        added.add(author_name)

    for a1, neighbors in coauthors_map.items():
        for a2, count in neighbors.items():
            if a2 in added and a1 < a2:  # evitare duplicati
                edges.append({
                    "from": a1,
                    "to": a2,
                    "label": str(count)
                })

    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
