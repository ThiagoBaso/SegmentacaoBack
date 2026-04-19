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

from app import app
