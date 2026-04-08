import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.logging_config import log
from app.services.image_processing import calcular_area_pixels
from app.services.sam_service import segmentar_ponto
from app.state import state


router = APIRouter()


@router.websocket("/ws/{session_id}")
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
    """
    if session_id not in state.sessoes:
        await websocket.close(code=4004, reason="Sessão não encontrada. Faça o upload primeiro.")
        return

    await websocket.accept()
    log.info(f"WebSocket conectado: {session_id}")
    sessao = state.sessoes[session_id]

    try:
        while True:
            raw = await websocket.receive_text()
            dados = json.loads(raw)
            acao = dados.get("acao")

            if acao == "ponto":
                print("ponto recebido")

                x = int(dados["x"])
                y = int(dados["y"])
                label = int(dados.get("label", 1))

                sessao["pontos_atuais"].append({"x": x, "y": y, "label": label})

                loop = asyncio.get_event_loop()
                resultado = await loop.run_in_executor(
                    None,
                    segmentar_ponto,
                    sessao["imagem_rgb"],
                    sessao["pontos_atuais"],
                )

                if resultado is None:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Nenhuma região detectada nesse ponto. Tente clicar em outro local.",
                    })
                else:
                    await websocket.send_json({
                        "tipo": "preview",
                        "poligono": resultado["poligono"],
                        "area_pixels": resultado["area_pixels"],
                        "score": resultado["score"],
                        "pontos": sessao["pontos_atuais"],
                    })

            elif acao == "confirmar":
                if not sessao["pontos_atuais"]:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Nenhum ponto ativo para confirmar.",
                    })
                    continue

                loop = asyncio.get_event_loop()
                resultado = await loop.run_in_executor(
                    None,
                    segmentar_ponto,
                    sessao["imagem_rgb"],
                    sessao["pontos_atuais"],
                )

                if resultado is None:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Não foi possível confirmar: talhão inválido.",
                    })
                    continue

                novo_id = len(sessao["talhoes"]) + 1
                talhao = {
                    "id": novo_id,
                    "poligono": resultado["poligono"],
                    "area_pixels": resultado["area_pixels"],
                }
                sessao["talhoes"].append(talhao)
                sessao["pontos_atuais"] = []

                log.info(f"Sessão {session_id} | Talhão {novo_id} confirmado | Área: {talhao['area_pixels']:.0f}px")

                await websocket.send_json({
                    "tipo": "talhao_confirmado",
                    "id": novo_id,
                    "poligono": talhao["poligono"],
                    "area_pixels": talhao["area_pixels"],
                    "todos_talhoes": sessao["talhoes"],
                })

            elif acao == "desfazer":
                if not sessao["talhoes"]:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": "Nenhum talhão para desfazer.",
                    })
                    continue

                removido = sessao["talhoes"].pop()
                log.info(f"Sessão {session_id} | Talhão {removido['id']} removido")

                await websocket.send_json({
                    "tipo": "desfeito",
                    "id_removido": removido["id"],
                    "todos_talhoes": sessao["talhoes"],
                })

            elif acao == "reiniciar":
                sessao["pontos_atuais"] = []
                await websocket.send_json({"tipo": "reiniciado"})

            elif acao == "editar_poligono":
                talhao_id = int(dados["id"])
                novo_poligono = dados["poligono"]

                talhao = next((item for item in sessao["talhoes"] if item["id"] == talhao_id), None)
                if talhao is None:
                    await websocket.send_json({
                        "tipo": "erro",
                        "mensagem": f"Talhão {talhao_id} não encontrado.",
                    })
                    continue

                talhao["poligono"] = novo_poligono
                talhao["area_pixels"] = calcular_area_pixels(novo_poligono)

                await websocket.send_json({
                    "tipo": "poligono_editado",
                    "id": talhao_id,
                    "area_pixels": talhao["area_pixels"],
                })

            else:
                await websocket.send_json({
                    "tipo": "erro",
                    "mensagem": f"Ação desconhecida: {acao}",
                })

    except WebSocketDisconnect:
        log.info(f"WebSocket desconectado: {session_id}")
    except Exception as exc:
        log.error(f"Erro na sessão {session_id}: {exc}")
        await websocket.send_json({"tipo": "erro", "mensagem": str(exc)})

