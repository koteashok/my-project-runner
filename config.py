"""
config.py
=========
Central configuration for Phase 1 of the
Emotion-Aware Multilingual Hybrid Recommender System.

This phase covers: dataset loading, schema inspection, preprocessing,
filtering, encoding, interaction-matrix construction and train/val/test
splitting.

DATASET_SOURCE selects the backbone:
    "google_local" -> Google Local Reviews (McAuley Lab, UCSD)   [default now]
    "amazon_c4"    -> McAuley-Lab/Amazon-C4 (HuggingFace)


"""

from __future__ import annotations
import os

# ----------------------------------------------------------------------------- #
# Which dataset backbone to use                                                  #
# ----------------------------------------------------------------------------- #
DATASET_SOURCE: str = "google_local"        # "google_local" | "amazon_c4"

# ---- Amazon-C4 (kept available) --------------------------------------------- #
DATASET_NAME: str = "McAuley-Lab/Amazon-C4"
DATASET_CONFIG: str | None = None
LOAD_ALL_SPLITS: bool = True

# ---- Google Local Reviews (McAuley Lab, UCSD) ------------------------------- #
# Version "2021" is United-States-only (50 states, up to Sep 2021) with the modern
#   schema: user_id / gmap_id / time / rating / text.  Best for dense CF + emotion
#   + rating variance, but English-dominant (Spanish/other minority).
# Version "2018" is the older GLOBAL dump (genuinely multilingual) with the schema:
#   gPlusUserId / gPlusPlaceId / unixReviewTime / rating / reviewText / categories.
#   Use it when balanced-ish multilingual text matters more; set its URL/PATH below.
GOOGLE_LOCAL_VERSION: str = "2021"          # "2021" (US) | "2018" (global/multilingual)

# Official mirrors of the dataset repository (either works):
#   https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/
#   https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/
GOOGLE_LOCAL_BASE_URL: str = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/"

# For the 2021 per-state files. State names with spaces use underscores
# (e.g. "New_York"). Verify the exact filename on the dataset page for your state.
GOOGLE_LOCAL_STATE: str = "California"

# Explicit overrides. If REVIEW_PATH points at an already-downloaded local
# .json.gz it is used directly (no network). Otherwise REVIEW_URL is streamed;
# if REVIEW_URL is empty it is constructed from BASE_URL + version/state.
GOOGLE_LOCAL_REVIEW_PATH: str | None = None
GOOGLE_LOCAL_REVIEW_URL: str | None = None

# The full US / global review files are enormous (hundreds of millions of rows).
# We stream and stop after this many raw rows so Phase 1 stays tractable; the
# MAX_SAMPLES cap below then subsamples further. Raise for a bigger experiment.
GOOGLE_LOCAL_MAX_RAW_ROWS: int = 300_000

# Optional category filter (best-effort). Requires the matching meta-<State>.json.gz
# to map business -> category. Leave None to skip (recommended for a first run).
GOOGLE_LOCAL_CATEGORY: str | None = None
GOOGLE_LOCAL_META_PATH: str | None = None
GOOGLE_LOCAL_META_URL: str | None = None
GOOGLE_LOCAL_META_MAX_ROWS: int = 2_000_000
NETWORK_TIMEOUT_SECONDS: int = 60

# ----------------------------------------------------------------------------- #
# Core research parameters                                                       #
# ----------------------------------------------------------------------------- #
MIN_USER_INTERACTIONS: int = 5
MIN_ITEM_INTERACTIONS: int = 5
MAX_SAMPLES: int = 100_000                  # cap on rows actually used (None = no cap)
RANDOM_SEED: int = 42
RATING_THRESHOLD: float = 4.0

# ----------------------------------------------------------------------------- #
# Interaction construction                                                       #
# ----------------------------------------------------------------------------- #
INTERACTION_MODE: str = "auto"              # "rating" | "implicit" | "auto"
DEDUPLICATE_INTERACTIONS: bool = True
REVIEW_KEEP_STRATEGY: str = "longest"       # "longest" | "first" | "last"

# ----------------------------------------------------------------------------- #
# Splitting                                                                      #
# ----------------------------------------------------------------------------- #
VALIDATION_ENABLED: bool = True
MIN_TRAIN_INTERACTIONS_FOR_EVAL: int = 1
DROP_COLD_ITEMS_IN_EVAL: bool = True

# ----------------------------------------------------------------------------- #
# Graceful handling of degenerate filtering                                      #
# ----------------------------------------------------------------------------- #
# Google Local (unlike Amazon-C4) has genuine repeat interactions, so a 5-core is
# usually fine. The relaxation guard remains as a safety net for tiny subsets.
RELAX_FILTER_IF_EMPTY: bool = True
RELAXED_MIN_INTERACTIONS: int = 1

# ----------------------------------------------------------------------------- #
# Text cleaning                                                                  #
# ----------------------------------------------------------------------------- #
UNICODE_NORMALIZATION_FORM: str = "NFKC"
LOWERCASE_TEXT: bool = False
NOISE_TOKEN_PATTERNS = [
    r"\[\[VIDEOID:[^\]]*\]\]",
    r"\[\[IMAGEID:[^\]]*\]\]",
    r"\[\[[A-Z]+ID:[^\]]*\]\]",
]

# ----------------------------------------------------------------------------- #
# Output locations                                                               #
# ----------------------------------------------------------------------------- #
RESULTS_DIR: str = "results"
PROCESSED_DIR: str = os.path.join(RESULTS_DIR, "processed")
MAPPINGS_DIR: str = os.path.join(RESULTS_DIR, "mappings")
STATS_PATH: str = os.path.join(RESULTS_DIR, "dataset_statistics.json")
WRITE_CSV: bool = True
WRITE_PARQUET: bool = True
N_SAMPLE_RECORDS: int = 5


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, PROCESSED_DIR, MAPPINGS_DIR):
        os.makedirs(d, exist_ok=True)


def resolve_google_local_review_url() -> str | None:
    """Build the review .json.gz URL from base/version/state if not overridden."""
    if GOOGLE_LOCAL_REVIEW_URL:
        return GOOGLE_LOCAL_REVIEW_URL
    if GOOGLE_LOCAL_VERSION == "2021":
        state = GOOGLE_LOCAL_STATE.strip().replace(" ", "_")
        base = GOOGLE_LOCAL_BASE_URL.rstrip("/")
        return f"{base}/review-{state}.json.gz"
    # 2018/global: no verified canonical filename -> require explicit override.
    return None


def as_dict() -> dict:
    keys = [
        "DATASET_SOURCE", "DATASET_NAME", "GOOGLE_LOCAL_VERSION",
        "GOOGLE_LOCAL_STATE", "GOOGLE_LOCAL_BASE_URL", "GOOGLE_LOCAL_REVIEW_PATH",
        "GOOGLE_LOCAL_REVIEW_URL", "GOOGLE_LOCAL_MAX_RAW_ROWS",
        "GOOGLE_LOCAL_CATEGORY", "MIN_USER_INTERACTIONS", "MIN_ITEM_INTERACTIONS",
        "MAX_SAMPLES", "RANDOM_SEED", "RATING_THRESHOLD", "INTERACTION_MODE",
        "DEDUPLICATE_INTERACTIONS", "REVIEW_KEEP_STRATEGY", "VALIDATION_ENABLED",
        "MIN_TRAIN_INTERACTIONS_FOR_EVAL", "DROP_COLD_ITEMS_IN_EVAL",
        "RELAX_FILTER_IF_EMPTY", "RELAXED_MIN_INTERACTIONS",
        "UNICODE_NORMALIZATION_FORM", "LOWERCASE_TEXT",
    ]
    g = globals()
    return {k: g.get(k) for k in keys}


# ============================================================================= #
# Phase 2 — LightGCN collaborative filtering                                    #
# (additive; does not modify the Phase 1 pipeline)                              #
# ============================================================================= #
EMBEDDING_DIM: int = 64
NUM_LAYERS: int = 3                 # K in the propagation; final = mean of layers 0..K
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 1e-5          # Adam weight_decay
BPR_REG_LAMBDA: float = 1e-4        # explicit L2 on ego (layer-0) embeddings = lambda in the BPR objective
EPOCHS: int = 20
BATCH_SIZE: int = 1024
NUM_NEG: int = 1                    # negatives per positive for BPR
EMB_INIT_STD: float = 0.1          # std for normal init of E^(0)

# Evaluation / model selection
TOP_K = [10, 20]                   # cutoffs reported
EVAL_K: int = 20                   # cutoff used for early-stopping / model selection
EVAL_METRIC: str = "recall"        # "recall" | "ndcg"
EARLY_STOP_PATIENCE: int = 5
EVAL_EVERY: int = 1
EVAL_USER_BATCH: int = 1024        # users scored per block during full-ranking eval

# Artefacts
CHECKPOINT_DIR: str = os.path.join(RESULTS_DIR, "checkpoints")
LIGHTGCN_BEST_PATH: str = os.path.join(CHECKPOINT_DIR, "lightgcn_best.pt")
LIGHTGCN_HISTORY_PATH: str = os.path.join(RESULTS_DIR, "lightgcn_training_history.csv")


def ensure_checkpoint_dir() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================================= #
# Phase 3 — XLM-RoBERTa multilingual review representation & sentiment           #
# (additive; does not modify Phase 1 preprocessing or Phase 2 LightGCN)          #
# ============================================================================= #
XLMR_MODEL_NAME: str = "xlm-roberta-base"   # do NOT swap for an English-only BERT
XLMR_MAX_LENGTH: int = 256
XLMR_EXTRACT_BATCH_SIZE: int = 32           # batch size for frozen feature extraction
XLMR_HIDDEN: int = 768                      # xlm-roberta-base hidden size (verified from model at runtime)
FINE_TUNE_ENCODER: bool = False             # default: freeze XLM-R, train only the head
USE_MIXED_PRECISION: bool = True            # autocast fp16 when a CUDA GPU is present
CACHE_FP16: bool = True                     # store cached review embeddings as float16 (halves disk)

# --- rating-derived (WEAK) sentiment labels -- NOT human-annotated -----------
# Fully configurable thresholds. rating <= NEG_MAX -> negative; == NEU_VALUE -> neutral; else positive.
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_NUM_CLASSES: int = 3
SENTIMENT_NEG_MAX: float = 2.0              # 1-2 -> negative
SENTIMENT_NEU_VALUE: float = 3.0           # 3   -> neutral
# (>= 4 -> positive)
SENTIMENT_REQUIRE_TEXT: bool = True         # only train/eval sentiment on rows with non-empty review

# --- sentiment head training -------------------------------------------------
SENTIMENT_EPOCHS: int = 20
SENTIMENT_LR: float = 1e-3
SENTIMENT_WEIGHT_DECAY: float = 1e-5
SENTIMENT_BATCH_SIZE: int = 256
SENTIMENT_PATIENCE: int = 5
SENTIMENT_CLASS_WEIGHTING: bool = True      # weight CE by inverse class frequency (labels are imbalanced)
SENTIMENT_SELECT_METRIC: str = "macro_f1"   # model-selection metric on validation

# --- artefacts ---------------------------------------------------------------
CACHE_DIR: str = os.path.join(RESULTS_DIR, "cache")
REVIEW_EMB_CACHE: str = os.path.join(CACHE_DIR, "review_embeddings.pt")
USER_REVIEW_EMB_CACHE: str = os.path.join(CACHE_DIR, "user_review_emb.pt")
ITEM_REVIEW_EMB_CACHE: str = os.path.join(CACHE_DIR, "item_review_emb.pt")
XLMR_SENTIMENT_BEST: str = os.path.join(CHECKPOINT_DIR, "xlmr_sentiment_best.pt")
XLMR_SENTIMENT_RESULTS: str = os.path.join(RESULTS_DIR, "xlmr_sentiment_results.csv")
XLMR_SENTIMENT_HISTORY: str = os.path.join(RESULTS_DIR, "xlmr_sentiment_history.json")


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================= #
# Phase 4 — Emotion representation (auxiliary, model-derived PSEUDO-LABELS)      #
# The dataset has NO human-annotated emotion labels (verified in Phase 1 schema  #
# inspection), so emotion features are generated by a pretrained multilingual    #
# classifier. They are AUXILIARY features, NOT ground truth.                     #
# ============================================================================= #
# Default: XLM-RoBERTa-base, 23 languages, 11 emotions (multi-label / sigmoid).
# Labels are READ FROM model.config.id2label at runtime, never hardcoded.
# Verified multilingual alternatives you can drop in here:
#   "poltextlab/xlm-roberta-large-pooled-emotions9-v2"  (XLM-R-large, 9 labels, gated)
#   "AnasAlokla/multilingual_go_emotions_V1.2"          (mBERT, 28 GoEmotions labels)
EMOTION_MODEL_NAME: str = "tabularisai/multilingual-emotion-classification"
EMOTION_MAX_LENGTH: int = 192
EMOTION_EXTRACT_BATCH_SIZE: int = 32
# "auto" reads model.config.problem_type (multi_label -> sigmoid, else softmax).
EMOTION_ACTIVATION: str = "auto"            # "auto" | "softmax" | "sigmoid"
EMOTION_ENTROPY_EPS: float = 1e-12
# emotion "intensity" definition (documented, configurable):
#   "one_minus_neutral" -> 1 - P(neutral-like label) (falls back to max_prob if none)
#   "max_prob"          -> the maximum emotion probability
#   "peakedness"        -> 1 - normalized entropy (H / log C)
EMOTION_INTENSITY_MODE: str = "one_minus_neutral"

# --- z_ui = [ h_ui || e_ui || s_ui ] projection ------------------------------
# s_ui = sentiment class-probability vector (3-d) from the Phase 3 head ("probs")
#        or its raw logits ("logits").
SENTIMENT_REP: str = "probs"
FUSION_LATENT_DIM: int = 128                # configurable latent dimension
FUSION_USE_LAYERNORM: bool = True
FUSION_ACTIVATION: str = "gelu"             # "gelu" | "relu" | "none"

# --- artefacts ---------------------------------------------------------------
EMOTION_FEATURES_CACHE: str = os.path.join(CACHE_DIR, "emotion_features.pt")
EMOTION_REPR_CACHE: str = os.path.join(CACHE_DIR, "emotion_review_representation.pt")
EMOTION_STATS_CSV: str = os.path.join(RESULTS_DIR, "emotion_statistics.csv")
FIGURES_DIR: str = os.path.join(RESULTS_DIR, "figures")
EMOTION_FIG: str = os.path.join(FIGURES_DIR, "emotion_distribution.png")


def ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================= #
# Phase 5 — Hybrid fusion (LightGCN collaborative  +  XLM-R review-based)        #
# Actually COMBINES both representations (not comparison-only).                  #
# ============================================================================= #
# Fusion strategy:
#   "weighted"           -> h = ALPHA*x~ + (1-ALPHA)*z~   (fixed ALPHA)
#   "weighted_learnable" -> ALPHA is a learnable scalar (via a logit)
#   "gate"               -> h = g⊙x~ + (1-g)⊙z~,  g = σ(W_g[x~||z~]+b_g)
#   "concat"             -> h = W_o[x~ || z~]
FUSION_STRATEGY: str = "weighted"
ALPHA: float = 0.5                     # initial / fixed fusion weight
FUSION_COMMON_DIM: int = 64            # common projected latent dimension d_common
FUSION_PROJECTION_BIAS: bool = False   # W_c, W_t bias (formula has none)

# Text (review) user/item representation source — leakage-safe.
#   "train" -> aggregate ONLY train reviews into z_u / z_i (used for train AND eval)
REVIEW_REPR_SOURCE: str = "train"

# Initialise the collaborative branch from the Phase 2 LightGCN checkpoint if present.
HYBRID_INIT_LIGHTGCN_FROM_CKPT: bool = True

# --- joint objective ---------------------------------------------------------
# L = L_BPR + gamma*L_sent + eta*L_emotion + lambda*||Theta||^2
# Emotion labels are model-derived PSEUDO-LABELS, so they are used as INPUT
# FEATURES (inside z), NOT as a supervised loss:  eta is pinned to 0.
HYBRID_SENTIMENT_AUX: bool = False     # optional gamma*L_sent (rating-derived weak labels)
HYBRID_GAMMA: float = 0.0
HYBRID_ETA: float = 0.0                # MUST stay 0 — emotion is features-only (pseudo-labels)
HYBRID_L2: float = 1e-4                # lambda on ego embeddings (BPR reg term)

# --- training ----------------------------------------------------------------
HYBRID_EPOCHS: int = 30
HYBRID_LR: float = 0.001
HYBRID_WEIGHT_DECAY: float = 1e-5      # Adam weight decay (covers projection/gate params)
HYBRID_BATCH_SIZE: int = 1024
HYBRID_NUM_NEG: int = 1
HYBRID_PATIENCE: int = 5
HYBRID_TOPK = [10, 20]
HYBRID_EVAL_K: int = 20
HYBRID_EVAL_METRIC: str = "recall"

# --- artefacts ---------------------------------------------------------------
HYBRID_BEST_PATH: str = os.path.join(CHECKPOINT_DIR, "hybrid_best.pt")
HYBRID_HISTORY_PATH: str = os.path.join(RESULTS_DIR, "hybrid_training_history.csv")


# ============================================================================= #
# Phase 6 — Experimental comparison of 5 recommenders                            #
# All models share the SAME train/val/test split and evaluation protocol.        #
# ============================================================================= #
K_VALUES = [5, 10, 20]
COMPARE_EPOCHS: int = 30
COMPARE_LR: float = 0.001
COMPARE_WEIGHT_DECAY: float = 1e-5
COMPARE_BATCH_SIZE: int = 1024
COMPARE_PATIENCE: int = 5
COMPARE_DIM: int = 64                 # latent dim for MF / common dim (fair, = EMBEDDING_DIM)
COMPARE_EVAL_K: int = 20              # metric cutoff used for early stopping
COMPARE_EVAL_METRIC: str = "recall"
MODEL_COMPARISON_CSV: str = os.path.join(RESULTS_DIR, "model_comparison.csv")


# ============================================================================= #
# Phase 7 — Ablation & sensitivity analysis                                     #
# Selection is done on VALIDATION NDCG@10; the test set is only reported.        #
# ============================================================================= #
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
LAYER_VALUES = [1, 2, 3, 4]              # LightGCN depth sweep (avoids clobbering NUM_LAYERS)
EMBEDDING_DIMS = [32, 64, 128]
SELECT_METRIC: str = "ndcg"             # model/hyperparameter selection metric ...
SELECT_K: int = 10                      # ... at this cutoff, on VALIDATION

ABLATION_RESULTS_CSV: str = os.path.join(RESULTS_DIR, "ablation_results.csv")
FUSION_SENS_CSV: str = os.path.join(RESULTS_DIR, "fusion_sensitivity.csv")
LAYER_SENS_CSV: str = os.path.join(RESULTS_DIR, "layer_sensitivity.csv")
EMBEDDING_SENS_CSV: str = os.path.join(RESULTS_DIR, "embedding_sensitivity.csv")
ABLATION_FIG_DIR: str = FIGURES_DIR


# ============================================================================= #
# Phase 8 — Final multilingual & emotion-aware analysis                          #
# ============================================================================= #
# If a language field exists it is used; otherwise language is INFERRED by a
# detector and clearly documented as inferred. No languages are assumed a priori.
LANG_DETECT_BACKEND: str = "langdetect"     # "langdetect" | "py3langid"
LANG_DETECT_SEED: int = 0
LANG_DETECT_MIN_CHARS: int = 1              # below this, label as "unknown"
LANG_DETECT_MAX_PER_SPLIT = None           # cap detections per split (None = all)
LANG_MIN_TEST_USERS: int = 20              # sufficiency for per-language rec metrics
EMOTION_MIN_TEST_USERS: int = 20           # sufficiency for per-emotion rec metrics

TABLES_DIR: str = os.path.join(RESULTS_DIR, "tables")
FINAL_RESULTS_DIR: str = os.path.join(RESULTS_DIR, "final_results")
LANG_RESULTS_CSV: str = os.path.join(RESULTS_DIR, "language_results.csv")
FINAL_COMPARISON_CSV: str = os.path.join(TABLES_DIR, "final_comparison.csv")
SENTIMENT_BY_LANG_CSV: str = os.path.join(TABLES_DIR, "sentiment_by_language.csv")
EMOTION_BY_LANG_CSV: str = os.path.join(TABLES_DIR, "emotion_by_language.csv")
EMOTION_BEHAVIOR_CSV: str = os.path.join(TABLES_DIR, "emotion_behavior.csv")
EMOTION_REC_CSV: str = os.path.join(TABLES_DIR, "emotion_recommendation.csv")
EXPERIMENT_SUMMARY_MD: str = os.path.join(FINAL_RESULTS_DIR, "experiment_summary.md")


def ensure_result_tree() -> None:
    for d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, CHECKPOINT_DIR, CACHE_DIR,
              FINAL_RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
