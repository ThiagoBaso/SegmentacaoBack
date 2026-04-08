from typing import Optional

import numpy as np

from app.state import state

from .image_processing import calcular_area_pixels, mascara_para_poligono, pos_processar_mascara


def segmentar_ponto(imagem_rgb: np.ndarray, pontos: list) -> Optional[dict]:
    """
    Roda o SAM com os pontos fornecidos e retorna o polígono.
    pontos = [{"x": int, "y": int, "label": 1 ou 0}, ...]
    label 1 = inclusão, 0 = exclusão
    """
    coords = np.array([[p["x"], p["y"]] for p in pontos])
    labels = np.array([p["label"] for p in pontos])

    masks, scores, _ = state.sam_predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
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

