import asyncio
import os
import uuid

import aiofiles
import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import MAX_SESSOES, PASTA_UPLOADS
from app.logging_config import log
from app.services.image_processing import preprocessar
from app.state import state


router = APIRouter()


@router.post("/upload")
async def upload_imagem(arquivo: UploadFile = File(...)):
    """
    Recebe a imagem, pré-processa, e seta no SAM predictor.
    Retorna session_id que será usado no WebSocket.

    Exemplo de uso no frontend:
        const form = new FormData()
        form.append("arquivo", file)
        const res = await fetch("http://servidor/upload", { method: "POST", body: form })
        const { session_id, largura, altura } = await res.json()
    """
    if len(state.sessoes) >= MAX_SESSOES:
        raise HTTPException(status_code=503, detail="Servidor com muitas sessões ativas. Tente novamente em instantes.")

    if not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem.")

    session_id = str(uuid.uuid4())
    caminho = os.path.join(PASTA_UPLOADS, f"{session_id}_{arquivo.filename}")

    async with aiofiles.open(caminho, "wb") as file_handle:
        await file_handle.write(await arquivo.read())

    imagem_bgr = cv2.imread(caminho)
    if imagem_bgr is None:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem enviada.")

    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    imagem_proc = preprocessar(imagem_rgb)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, state.sam_predictor.set_image, imagem_proc)

    state.sessoes[session_id] = {
        "imagem_rgb": imagem_rgb,
        "imagem_proc": imagem_proc,
        "imagem_path": caminho,
        "talhoes": [],
        "pontos_atuais": [],
    }

    log.info(f"Sessão criada: {session_id} | Imagem: {imagem_rgb.shape[1]}x{imagem_rgb.shape[0]}px")

    return {
        "session_id": session_id,
        "largura": imagem_rgb.shape[1],
        "altura": imagem_rgb.shape[0],
        "mensagem": "Imagem pronta. Conecte ao WebSocket para iniciar a seleção.",
    }


@router.get("/fazendas")
async def get_fazendas():
    return [
        {
            "id": "faz_001",
            "nome": "Fazenda Boa Vista",
            "thumbnail": None,
            "localizacao": {"cidade": "Tupã", "estado": "SP"},
            "area_total_ha": 320.5,
            "talhoes": [
                {"id": "tal_001", "nome": "Talhão A", "area_ha": 85.2, "cor": "#00FFAA"},
                {"id": "tal_002", "nome": "Talhão B", "area_ha": 112.0, "cor": "#00AAFF"},
            ],
        }
    ]


@router.get("/sessao/{session_id}")
async def info_sessao(session_id: str):
    """Retorna os talhões confirmados da sessão."""
    if session_id not in state.sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    sessao = state.sessoes[session_id]
    return {
        "session_id": session_id,
        "num_talhoes": len(sessao["talhoes"]),
        "talhoes": sessao["talhoes"],
    }


@router.delete("/sessao/{session_id}")
async def encerrar_sessao(session_id: str):
    """
    Encerra a sessão, libera memória e apaga a imagem do disco.
    Chame isso quando o usuário sair da página ou finalizar o trabalho.
    """
    if session_id not in state.sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    sessao = state.sessoes.pop(session_id)

    if os.path.exists(sessao["imagem_path"]):
        os.remove(sessao["imagem_path"])

    log.info(f"Sessão encerrada: {session_id}")
    return {"mensagem": "Sessão encerrada com sucesso."}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "sam_carregado": state.sam_predictor is not None,
        "sessoes_ativas": len(state.sessoes),
    }

