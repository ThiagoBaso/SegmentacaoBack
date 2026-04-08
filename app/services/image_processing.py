from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from app.config import AREA_MINIMA_PX, EPSILON_SIMPLIFICAR, KERNEL_MORFOLOGICO


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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

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

    contorno = max(contornos, key=cv2.contourArea)

    if cv2.contourArea(contorno) < AREA_MINIMA_PX:
        return None

    epsilon = EPSILON_SIMPLIFICAR * cv2.arcLength(contorno, True) / 100
    contorno = cv2.approxPolyDP(contorno, epsilon, True)

    return contorno.reshape(-1, 2).tolist()


def calcular_area_pixels(poligono: list) -> float:
    """Área do polígono em pixels²."""
    contorno = np.array(poligono).reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.contourArea(contorno))

