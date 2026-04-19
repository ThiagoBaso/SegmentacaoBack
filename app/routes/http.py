import asyncio
import os
import uuid

import aiofiles
import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import MAX_SESSOES, PASTA_UPLOADS
from app.logging_config import log
from app.services.image_processing import preprocessar
from app.services.geotiff_service import processar_geotiff
from app.state import state

router = APIRouter()

EXTENSOES_GEOTIFF = {".tif", ".tiff"}


def _eh_geotiff(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in EXTENSOES_GEOTIFF


@router.post("/upload")
async def upload_imagem(arquivo: UploadFile = File(...)):
    if len(state.sessoes) >= MAX_SESSOES:
        raise HTTPException(status_code=503, detail="Servidor com muitas sessões ativas.")

    # GeoTIFF pode chegar como application/octet-stream — valida pela extensão também
    eh_tiff = _eh_geotiff(arquivo.filename)
    if not eh_tiff and not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem.")

    session_id = str(uuid.uuid4())
    caminho = os.path.join(PASTA_UPLOADS, f"{session_id}_{arquivo.filename}")

    async with aiofiles.open(caminho, "wb") as file_handle:
        await file_handle.write(await arquivo.read())

    # --- Processamento separado por tipo ---
    bounds = None

    if eh_tiff:
        try:
            resultado = processar_geotiff(caminho)
            imagem_rgb = resultado["imagem_rgb"]
            bounds = resultado["bounds"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar GeoTIFF: {str(e)}")
    else:
        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            raise HTTPException(status_code=400, detail="Não foi possível ler a imagem enviada.")
        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    # --- Salva PNG convertido para servir ao frontend ---
    png_path = caminho.rsplit(".", 1)[0] + ".png"
    imagem_bgr_para_salvar = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(png_path, imagem_bgr_para_salvar)

    # --- Prepara para o SAM ---
    imagem_proc = preprocessar(imagem_rgb)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, state.sam_predictor.set_image, imagem_proc)

    state.sessoes[session_id] = {
        "imagem_rgb": imagem_rgb,
        "imagem_proc": imagem_proc,
        "imagem_path": png_path,   # aponta pro PNG convertido
        "bounds": bounds,
        "talhoes": [],
        "pontos_atuais": [],
    }

    log.info(f"Sessão criada: {session_id} | {imagem_rgb.shape[1]}x{imagem_rgb.shape[0]}px | GeoTIFF: {eh_tiff}")

    return {
        "session_id": session_id,
        "largura": imagem_rgb.shape[1],
        "altura": imagem_rgb.shape[0],
        "bounds": bounds,          # None se não georeferenciada
        "georeferenciada": eh_tiff,
        "mensagem": "Imagem pronta. Conecte ao WebSocket para iniciar a seleção.",
    }

@router.get("/imagem/{session_id}")
async def servir_imagem(session_id: str):

    print("GET IMAGEM")

    """Serve o PNG convertido para o frontend renderizar."""
    if session_id not in state.sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    png_path = state.sessoes[session_id]["imagem_path"]
    if not os.path.exists(png_path):
        raise HTTPException(status_code=404, detail="Imagem não encontrada.")

    return FileResponse(png_path, media_type="image/png")


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

