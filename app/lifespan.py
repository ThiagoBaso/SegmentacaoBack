import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import MODELO_PATH, PASTA_UPLOADS
from .logging_config import log
from .state import state


os.makedirs(PASTA_UPLOADS, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Carregando modelo SAM...")
    try:
        from segment_anything_hq import SamPredictor, sam_model_registry
        import torch

        if not os.path.exists(MODELO_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODELO_PATH}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Dispositivo: {device}")

        sam = sam_model_registry["vit_h"](checkpoint=MODELO_PATH)
        sam.to(device=device)
        state.sam_predictor = SamPredictor(sam)

        log.info("SAM carregado com sucesso!")
    except Exception as exc:
        log.error(f"Erro ao carregar SAM: {exc}")
        raise

    yield

    log.info("Encerrando servidor, limpando sessões...")
    state.sessoes.clear()

