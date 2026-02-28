"""
main.py
FastAPI server to test Collaborative and Content-Based Filtering engines.

Run with:
    uvicorn main:app --reload

First time setup:
    python save_models.py  ← run this once before starting the server
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import pickle
import os
import time

from collaborative import (
    build_engine,
    load_model,
    get_recommendations,
    enrich_with_product_details,
)
from content_engine import (
    load_content_model,
    get_content_recommendations,
    get_item_similarity,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Models Directory ──────────────────────────────────────────────────────────

MODELS_DIR = "models"


# ── App State (loaded once at startup) ───────────────────────────────────────

# ── App State (loaded on demand) ───────────────────────────────────────────

class AppState:
    engine        = None
    _matrix       = None
    _sim_matrix   = None
    _product_df   = None
    _tfidf_matrix = None
    _product_index = None

state = AppState()


# ── Lazy Loading ─────────────────────────────────────────────────────────────

def get_matrix():
    if state._matrix is None:
        logger.info("📂 Lazy-loading CF matrix ...")
        with open(f"{MODELS_DIR}/cf_matrix.pkl", "rb") as f:
            state._matrix = pickle.load(f)
    return state._matrix

def get_sim_matrix():
    if state._sim_matrix is None:
        logger.info("📂 Lazy-loading CF sim matrix ...")
        with open(f"{MODELS_DIR}/cf_sim_matrix.pkl", "rb") as f:
            state._sim_matrix = pickle.load(f)
    return state._sim_matrix

def get_product_df():
    if state._product_df is None:
        logger.info("📂 Lazy-loading product DF ...")
        with open(f"{MODELS_DIR}/product_df.pkl", "rb") as f:
            state._product_df = pickle.load(f)
    return state._product_df

def get_tfidf_matrix():
    if state._tfidf_matrix is None:
        logger.info("📂 Lazy-loading TF-IDF matrix ...")
        with open(f"{MODELS_DIR}/tfidf_matrix.pkl", "rb") as f:
            state._tfidf_matrix = pickle.load(f)
    return state._tfidf_matrix

def get_product_index():
    if state._product_index is None:
        logger.info("📂 Lazy-loading product index ...")
        with open(f"{MODELS_DIR}/product_index.pkl", "rb") as f:
            state._product_index = pickle.load(f)
    return state._product_index


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — Engine only (Lazy Loading enabled)")
    state.engine = build_engine()
    yield
    logger.info("Shutting down — disposing DB engine ...")
    state.engine.dispose()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Recommendation Engine API",
    description="Collaborative Filtering and Content-Based Filtering endpoints",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def validate_user(user_id: int):
    """Raise 404 if user doesn't exist in the rating matrix."""
    matrix = get_matrix()
    if user_id not in matrix.index:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found in the rating matrix."
        )


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health_check():
    """Check if the API is live."""
    return {
        "status": "ok",
        "engine_live": state.engine is not None,
        "memory_optimized": True
    }


# ── Endpoint 1: Collaborative Filtering ──────────────────────────────────────

@app.get("/recommend/collaborative/{user_id}", tags=["Collaborative Filtering"])
def collaborative_recommendations(
    user_id : int,
    top_n   : int  = Query(default=10, ge=1, le=50,  description="Number of recommendations to return"),
    enrich  : bool = Query(default=True,              description="Include full product details"),
):
    validate_user(user_id)
    rec_df = get_recommendations(user_id, get_matrix(), get_sim_matrix(), top_n=top_n)
    if rec_df.empty:
        return JSONResponse(content={"user_id": user_id, "recommendations": [], "count": 0})
    if enrich:
        rec_df = enrich_with_product_details(rec_df, state.engine)
    return {
        "user_id": user_id,
        "recommendations": rec_df.to_dict(orient="records"),
    }


# ── Endpoint 2: Content-Based Filtering ──────────────────────────────────────

@app.get("/recommend/content/{user_id}", tags=["Content-Based Filtering"])
def content_recommendations(
    user_id     : int,
    top_n       : int  = Query(default=10, ge=1, le=50, description="Number of recommendations to return"),
    top_rated_n : int  = Query(default=5,  ge=1, le=20, description="User's top-rated products to use as taste seeds"),
    enrich      : bool = Query(default=True,             description="Include full product details"),
):
    validate_user(user_id)
    rec_df = get_content_recommendations(
        user_id, get_matrix(), get_tfidf_matrix(), get_product_index(), top_n=top_n, top_rated_n=top_rated_n
    )
    if rec_df.empty:
        return JSONResponse(content={"user_id": user_id, "recommendations": [], "count": 0})
    if enrich:
        rec_df = enrich_with_product_details(rec_df, state.engine)
    return {
        "user_id": user_id,
        "recommendations": rec_df.to_dict(orient="records"),
    }


# ── Endpoint 3: Search ────────────────────────────────────────────────────────

@app.get("/search", tags=["Search"])
def search_products(
    q          : str   = Query(..., min_length=1),
    top_n      : int   = Query(default=10, ge=1, le=50),
    min_rating : float = Query(default=0.0, ge=0.0, le=5.0),
):
    try:
        from sqlalchemy import text
        query = "SELECT asin, title, stars, reviews, price, img_url AS imgUrl FROM amazon_products WHERE title LIKE :q AND stars >= :r ORDER BY stars DESC LIMIT :n"
        with state.engine.connect() as conn:
            rows = conn.execute(text(query), {"q": f"%{q}%", "r": min_rating, "n": top_n}).fetchall()
        return {"items": [dict(r._mapping) for r in rows]}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 4: Homepage Paginated Products ───────────────────────────────────

@app.get("/products", tags=["Compatibility"])
@app.get("/products/homepage", tags=["Compatibility"])
def get_products_compat(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str = Query(None)
):
    try:
        from sqlalchemy import text
        offset = (page - 1) * size
        where_clause = "WHERE title IS NOT NULL"
        params = {"limit": size, "offset": offset}
        if search:
            where_clause += " AND title LIKE :search"
            params["search"] = f"%{search}%"
            
        # Optimization: Remove COUNT(*) if possible, but keeping it for pagination
        query = f"SELECT asin, title, stars, reviews, price, img_url AS imgUrl FROM amazon_products {where_clause} LIMIT :limit OFFSET :offset"
        count_query = f"SELECT COUNT(*) FROM amazon_products {where_clause}"
        
        with state.engine.connect() as conn:
            rows = conn.execute(text(query), params).fetchall()
            # Fast-path for count if it's the first page
            total_count = conn.execute(text(count_query), params).scalar()
            
        return {"items": [dict(r._mapping) for r in rows], "pages": (total_count + size - 1) // size}
    except Exception as e:
        logger.error(f"Products error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{asin}", tags=["Compatibility"])
def get_product_detail_compat(asin: str):
    try:
        from sqlalchemy import text
        query = "SELECT asin, title, stars, reviews, price, img_url AS imgUrl, video_url FROM amazon_products WHERE asin = :asin"
        with state.engine.connect() as conn:
            row = conn.execute(text(query), {"asin": asin}).fetchone()
        if not row: raise HTTPException(status_code=404)
        return dict(row._mapping)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{asin}/similar", tags=["Compatibility"])
def get_similar_products_compat(asin: str):
    try:
        rec_df = get_item_similarity(asin, get_tfidf_matrix(), get_product_index(), top_n=6)
        if rec_df.empty: return []
        enriched = enrich_with_product_details(rec_df, state.engine)
        return [{"similar_product": dict(row._mapping)} for row in enriched.itertuples()]
    except Exception as e:
        return []


@app.get("/users/{user_id}/recommendations", tags=["Compatibility"])
def get_user_recommendations_compat(user_id: int):
    try:
        cf_df = get_recommendations(user_id, get_matrix(), get_sim_matrix(), top_n=5)
        if cf_df.empty: return []
        from sqlalchemy import text
        query = "SELECT asin, title, stars, reviews, price, img_url AS imgUrl FROM amazon_products WHERE asin IN :ids"
        with state.engine.connect() as conn:
            rows = conn.execute(text(query), {"ids": tuple(cf_df["product_id"].tolist())}).fetchall()
        return [{"product": dict(r._mapping)} for r in rows]
    except Exception as e:
        return []