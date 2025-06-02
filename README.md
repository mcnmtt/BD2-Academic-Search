<p align="center">
    <img width="200" src="https://www.opisalerno.it/wp-content/uploads/2016/11/logo-unisa-png.png" alt="Logo UNISA">
</p>
<h2 align="center">
PROGETTO BASI DI DATI 2

prof.ssa Tortora, prof. Di Biasi
<h1 align="center"> BD² - Academic Search <h2 align="center">

**BD²** è un portale di ricerca accademica sviluppato per il corso di *Basi di Dati 2*, progettato per esplorare e visualizzare pubblicazioni scientifiche (es. da arXiv), con funzionalità avanzate di ricerca testuale e raccomandazione basata su similarità semantica tra paper.
---
## 🚀 Funzionalità principali

- 🔍 **Ricerca full-text** per titolo, autori, abstract e categorie.
- 📄 **Visualizzazione risultati** con anteprima dell’abstract, link al PDF arXiv e DOI.
- 🧠 **Raccomandazione di paper simili** tramite algoritmo HNSW su embeddings.
- 🔐 **Accesso amministratore** con login protetto da password hashata (bcrypt).
- ✏️ **Modifica ed eliminazione dei paper** da interfaccia amministrativa.
- ➕ **Inserimento di nuovi paper** con calcolo embedding automatico e aggiornamento HNSW.
- 📦 **Sistema dinamico** con MongoDB e indexing aggiornabile senza rigenerazione totale.
- 📁 **Interfaccia responsive** con Bootstrap 5.

---

## 🛠️ Tecnologie utilizzate

| Componente              | Tecnologia                       |
|------------------------|----------------------------------|
| Backend                | Flask (Python)                   |
| Frontend               | HTML + CSS + Bootstrap 5         |
| Database               | MongoDB                          |
| Embedding              | `sentence-transformers`          |
| Similarità vettoriale  | `hnswlib`                         |
| Sicurezza login        | `bcrypt`                         |
| Indicizzazione         | `hnswlib` (cosine similarity)    |
| Visualizzazione        | Flask Templates (Jinja2)         |
| Deployment             | `Flask` on `localhost:5000`      |

---

## 📁 Struttura del progetto

```
bd2-academic-search/
├── app.py                  # Main Flask app
├── hnsw_index/             # Script e file per generare/aggiornare l'indice HNSW
│   ├── build_hnsw_index.py # Script per la creazione dell'indice
│   ├── hnsw_papers.index   # Indice vettoriale HNSW
│   └── paper_ids.npy       # Array ID dei paper (per HNSW)
├── templates/              # Template HTML (home, results, edit, partials)
├── static/                 # Logo, favicon e CSS
├── requirements.txt        # Dipendenze Python installabili con pip
├── environment.yml         # Ambiente Conda per sviluppo
└── README.md               
```

---

## ⚙️ Setup locale

1. **Clona la repository:**

```bash
git clone https://github.com/mcnmtt/BD2-Academic-Search.git
cd bd2-academic-search
```

2. **Crea ambiente Conda:**

```bash
conda env create -f environment.yml
conda activate bd2
```

3. **Avvia il server Flask:**

```bash
python app.py
```

4. **Accesso Admin:**  
   Crea manualmente un utente nel database MongoDB nella collection `users`, con password hashata usando `bcrypt`.

---

## 🔍 Come funziona il sistema di raccomandazione?

- Ogni paper ha un embedding (vettore) calcolato usando `SentenceTransformer (all-MiniLM-L6-v2)`.
- Gli embeddings sono indicizzati con **HNSW** (`hnswlib`) per permettere ricerche di similarità veloci.
- Quando un paper viene cercato o visualizzato, si possono calcolare e mostrare i paper più simili.

---

## ✍️ Autori

- **Mattia Maucioni**
- **Antonio Landi**
- Università degli Studi di Salerno  
  *Data Science & Machine Learning – Corso di Laurea Magistrale in Informatica*

---

## 📄 Licenza

Questo progetto è realizzato esclusivamente per scopi didattici e non ha scopi commerciali. Tutti i dati utilizzati sono a solo scopo di testing.

---
