"""
API SEGMENTADOR DE TALHÕES
===========================

INSTALAÇÃO:
    pip install fastapi uvicorn websockets python-multipart
    pip install opencv-python numpy Pillow torch torchvision segment-anything aiofiles

RODAR:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

ENDPOINTS:
    POST /upload                  → envia imagem, recebe session_id
    WS   /ws/{session_id}         → WebSocket para cliques em tempo real
    GET  /sessao/{session_id}     → info da sessão atual
    DELETE /sessao/{session_id}   → encerra sessão e libera memória
"""

import os
import uuid
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────

MODELO_PATH       = "sam_vit_h_4b8939.pth"
PASTA_UPLOADS     = "uploads"
AREA_MINIMA_PX    = 2000
KERNEL_MORFOLOGICO = 7
EPSILON_SIMPLIFICAR = 1.5        # simplificação Douglas-Peucker
MAX_SESSOES       = 20           # máximo de sessões simultâneas em memória

os.makedirs(PASTA_UPLOADS, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO GLOBAL — modelo e sessões
# ──────────────────────────────────────────────────────────────────────────────

# O modelo é carregado UMA VEZ quando o servidor sobe e fica em memória
sam_predictor = None

# Cada sessão guarda: imagem processada, predictor já setado, e talhões confirmados
# { session_id: { "imagem_rgb": np.array, "talhoes": [mask, ...], "imagem_path": str } }
sessoes: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP — carrega SAM antes de aceitar requisições
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam_predictor
    log.info("Carregando modelo SAM...")
    try:
        from segment_anything import SamPredictor, sam_model_registry
        import torch

        if not os.path.exists(MODELO_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODELO_PATH}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Dispositivo: {device}")

        sam = sam_model_registry["vit_h"](checkpoint=MODELO_PATH)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)

        log.info("SAM carregado com sucesso!")
    except Exception as e:
        log.error(f"Erro ao carregar SAM: {e}")
        raise

    yield  # servidor rodando

    # Shutdown
    log.info("Encerrando servidor, limpando sessões...")
    sessoes.clear()


app = FastAPI(title="Segmentador de Talhões", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # em produção, coloque a URL do seu frontend aqui
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────────────────────────────────────

def preprocessar(imagem_rgb: np.ndarray) -> np.ndarray:
    """CLAHE + sharpening antes de passar ao SAM."""
    img_lab = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    imagem_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    pil = Image.fromarray(imagem_rgb)
    pil = ImageEnhance.Sharpness(pil).enhance(1.8)
    return np.array(pil)


def pos_processar_mascara(mask: np.ndarray) -> np.ndarray:
    """Fecha buracos, suaviza bordas, remove ruídos."""
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_MORFOLOGICO, KERNEL_MORFOLOGICO)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    mask_f = mask.astype(np.float32)
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), 1.5)
    return (mask_f > 0.5).astype(np.uint8)


def mascara_para_poligono(mask: np.ndarray) -> Optional[list]:
    """
    Converte máscara binária em polígono simplificado.
    Retorna lista de pontos [x, y] ou None se inválido.
    """
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None

    # Pega o maior contorno
    contorno = max(contornos, key=cv2.contourArea)

    if cv2.contourArea(contorno) < AREA_MINIMA_PX:
        return None

    # Simplificação Douglas-Peucker
    epsilon = EPSILON_SIMPLIFICAR * cv2.arcLength(contorno, True) / 100
    contorno = cv2.approxPolyDP(contorno, epsilon, True)

    return contorno.reshape(-1, 2).tolist()


def calcular_area_pixels(poligono: list) -> float:
    """Área do polígono em pixels²."""
    contorno = np.array(poligono).reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.contourArea(contorno))


def segmentar_ponto(imagem_rgb: np.ndarray, pontos: list) -> Optional[dict]:
    """
    Roda o SAM com os pontos fornecidos e retorna o polígono.
    pontos = [{"x": int, "y": int, "label": 1 ou 0}, ...]
    label 1 = inclusão, 0 = exclusão
    """
    coords = np.array([[p["x"], p["y"]] for p in pontos])
    labels = np.array([p["label"] for p in pontos])

    # set_image é caro — só chama se a imagem mudou (controlado pela sessão)
    masks, scores, _ = sam_predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True
    )

    melhor_mask = masks[np.argmax(scores)]
    mask_processada = pos_processar_mascara(melhor_mask.astype(np.uint8))
    poligono = mascara_para_poligono(mask_processada)

    if poligono is None:
        return None

    return {
        "poligono": poligono,
        "area_pixels": calcular_area_pixels(poligono),
        "score": float(scores.max()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — UPLOAD DE IMAGEM
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
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
    if len(sessoes) >= MAX_SESSOES:
        raise HTTPException(status_code=503, detail="Servidor com muitas sessões ativas. Tente novamente em instantes.")

    # Valida tipo
    if not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem.")

    session_id = str(uuid.uuid4())
    caminho = os.path.join(PASTA_UPLOADS, f"{session_id}_{arquivo.filename}")

    # Salva imagem no disco
    async with aiofiles.open(caminho, "wb") as f:
        await f.write(await arquivo.read())

    # Carrega e pré-processa
    imagem_bgr = cv2.imread(caminho)
    if imagem_bgr is None:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem enviada.")

    imagem_rgb  = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    imagem_proc = preprocessar(imagem_rgb)

    # Seta a imagem no SAM (operação pesada — feita aqui, uma vez por sessão)
    # Isso roda em thread separada para não bloquear o event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sam_predictor.set_image, imagem_proc)

    # Guarda sessão
    sessoes[session_id] = {
        "imagem_rgb":   imagem_rgb,
        "imagem_proc":  imagem_proc,
        "imagem_path":  caminho,
        "talhoes":      [],          # talhões confirmados
        "pontos_atuais": [],         # pontos do talhão em edição
    }

    log.info(f"Sessão criada: {session_id} | Imagem: {imagem_rgb.shape[1]}x{imagem_rgb.shape[0]}px")

    return {
        "session_id": session_id,
        "largura":    imagem_rgb.shape[1],
        "altura":     imagem_rgb.shape[0],
        "mensagem":   "Imagem pronta. Conecte ao WebSocket para iniciar a seleção."
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — WEBSOCKET (comunicação em tempo real)
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_segmentacao(websocket: WebSocket, session_id: str):
    """
    WebSocket para comunicação em tempo real com o SAM.

    MENSAGENS QUE O FRONTEND ENVIA (JSON):

    1. Adicionar ponto (clique esquerdo ou direito):
        { "acao": "ponto", "x": 320, "y": 210, "label": 1 }
        label: 1 = incluir área, 0 = excluir área

    2. Confirmar talhão atual:
        { "acao": "confirmar" }

    3. Desfazer último talhão confirmado:
        { "acao": "desfazer" }

    4. Reiniciar pontos do talhão em edição:
        { "acao": "reiniciar" }

    5. Editar polígono manualmente (o usuário arrastou um ponto no Leaflet):
        { "acao": "editar_poligono", "id": 2, "poligono": [[x1,y1],[x2,y2],...] }

    ─────────────────────────────────────────────────────────────
    RESPOSTAS QUE O SERVIDOR ENVIA (JSON):

    Após "ponto":
        {
          "tipo": "preview",
          "poligono": [[x,y], ...],   ← contorno do talhão em edição
          "area_pixels": 45230,
          "score": 0.97
        }

    Após "confirmar":
        {
          "tipo": "talhao_confirmado",
          "id": 3,
          "poligono": [[x,y], ...],
          "area_pixels": 45230,
          "todos_talhoes": [ { "id": 1, ... }, { "id": 2, ... }, { "id": 3, ... } ]
        }

    Após "desfazer":
        {
          "tipo": "desfeito",
          "id_removido": 3,
          "todos_talhoes": [ { "id": 1, ... }, { "id": 2, ... } ]
        }

    Após "reiniciar":
        { "tipo": "reiniciado" }

    Após "editar_poligono":
        {
          "tipo": "poligono_editado",
          "id": 2,
          "area_pixels": 41000
        }

    Em caso de erro:
        { "tipo": "erro", "mensagem": "descrição do erro" }
    """

    if session_id not in sessoes:
        await websocket.close(code=4004, reason="Sessão não encontrada. Faça o upload primeiro.")
        return

    await websocket.accept()
    log.info(f"WebSocket conectado: {session_id}")
    sessao = sessoes[session_id]

    try:
        while True:
            raw = await websocket.receive_text()
            dados = json.loads(raw)
            acao  = dados.get("acao")

            # ── PONTO: usuário clicou na imagem ──────────────────────────────
            if acao == "ponto":
                print('ponto recebido')

                x     = int(dados["x"])
                y     = int(dados["y"])
                label = int(dados.get("label", 1))

                sessao["pontos_atuais"].append({"x": x, "y": y, "label": label})

                # Roda SAM em thread separada (não bloqueia o event loop)
                loop = asyncio.get_event_loop()
                resultado = await loop.run_in_executor(
                    None,
                    segmentar_ponto,
                    sessao["imagem_rgb"],
                    sessao["pontos_atuais"]
                )

                if resultado is None:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Nenhuma região detectada nesse ponto. Tente clicar em outro local."
                    })
                else:
                    await websocket.send_json({
                        "tipo":        "preview",
                        "poligono":    resultado["poligono"],
                        "area_pixels": resultado["area_pixels"],
                        "score":       resultado["score"],
                        "pontos":      sessao["pontos_atuais"],
                    })

            # ── CONFIRMAR: salva o talhão atual ──────────────────────────────
            elif acao == "confirmar":
                if not sessao["pontos_atuais"]:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Nenhum ponto ativo para confirmar."
                    })
                    continue

                loop = asyncio.get_event_loop()
                resultado = await loop.run_in_executor(
                    None,
                    segmentar_ponto,
                    sessao["imagem_rgb"],
                    sessao["pontos_atuais"]
                )

                if resultado is None:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Não foi possível confirmar: talhão inválido."
                    })
                    continue

                novo_id = len(sessao["talhoes"]) + 1
                talhao = {
                    "id":          novo_id,
                    "poligono":    resultado["poligono"],
                    "area_pixels": resultado["area_pixels"],
                }
                sessao["talhoes"].append(talhao)
                sessao["pontos_atuais"] = []   # limpa pontos para próximo talhão

                log.info(f"Sessão {session_id} | Talhão {novo_id} confirmado | Área: {talhao['area_pixels']:.0f}px")

                await websocket.send_json({
                    "tipo":             "talhao_confirmado",
                    "id":               novo_id,
                    "poligono":         talhao["poligono"],
                    "area_pixels":      talhao["area_pixels"],
                    "todos_talhoes":    sessao["talhoes"],
                })

            # ── DESFAZER: remove último talhão confirmado ─────────────────────
            elif acao == "desfazer":
                if not sessao["talhoes"]:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Nenhum talhão para desfazer."
                    })
                    continue

                removido = sessao["talhoes"].pop()
                log.info(f"Sessão {session_id} | Talhão {removido['id']} removido")

                await websocket.send_json({
                    "tipo":          "desfeito",
                    "id_removido":   removido["id"],
                    "todos_talhoes": sessao["talhoes"],
                })

            # ── REINICIAR: limpa pontos do talhão em edição ───────────────────
            elif acao == "reiniciar":
                sessao["pontos_atuais"] = []
                await websocket.send_json({"tipo": "reiniciado"})

            # ── EDITAR POLÍGONO: usuário arrastou ponto no Leaflet ────────────
            elif acao == "editar_poligono":
                talhao_id = int(dados["id"])
                novo_poligono = dados["poligono"]   # [[x,y], [x,y], ...]

                talhao = next((t for t in sessao["talhoes"] if t["id"] == talhao_id), None)
                if talhao is None:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": f"Talhão {talhao_id} não encontrado."
                    })
                    continue

                # Recalcula área com o novo polígono editado
                talhao["poligono"]    = novo_poligono
                talhao["area_pixels"] = calcular_area_pixels(novo_poligono)

                await websocket.send_json({
                    "tipo":        "poligono_editado",
                    "id":          talhao_id,
                    "area_pixels": talhao["area_pixels"],
                })

            else:
                await websocket.send_json({
                    "tipo": "erro",
                    "mensagem": f"Ação desconhecida: {acao}"
                })

    except WebSocketDisconnect:
        log.info(f"WebSocket desconectado: {session_id}")
    except Exception as e:
        log.error(f"Erro na sessão {session_id}: {e}")
        await websocket.send_json({"tipo": "erro", "mensagem": str(e)})


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — INFO DA SESSÃO
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/sessao/{session_id}")
async def info_sessao(session_id: str):
    """Retorna os talhões confirmados da sessão."""
    if session_id not in sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    sessao = sessoes[session_id]
    return {
        "session_id":   session_id,
        "num_talhoes":  len(sessao["talhoes"]),
        "talhoes":      sessao["talhoes"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — ENCERRAR SESSÃO
# ──────────────────────────────────────────────────────────────────────────────

@app.delete("/sessao/{session_id}")
async def encerrar_sessao(session_id: str):
    """
    Encerra a sessão, libera memória e apaga a imagem do disco.
    Chame isso quando o usuário sair da página ou finalizar o trabalho.
    """
    if session_id not in sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    sessao = sessoes.pop(session_id)

    # Apaga imagem do disco
    if os.path.exists(sessao["imagem_path"]):
        os.remove(sessao["imagem_path"])

    log.info(f"Sessão encerrada: {session_id}")
    return {"mensagem": "Sessão encerrada com sucesso."}


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — HEALTH CHECK
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "sam_carregado":   sam_predictor is not None,
        "sessoes_ativas":  len(sessoes),
    }