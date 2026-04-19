import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from pyproj import CRS

def processar_geotiff(caminho: str) -> dict:
    with rasterio.open(caminho) as src:
        crs_origem = src.crs
        bounds_src = src.bounds

        if crs_origem and crs_origem.to_epsg() != 4326:
            lng_min, lat_min, lng_max, lat_max = transform_bounds(
                crs_origem,
                CRS.from_epsg(4326),
                bounds_src.left,
                bounds_src.bottom,
                bounds_src.right,
                bounds_src.top,
            )
        else:
            lng_min = bounds_src.left
            lat_min = bounds_src.bottom
            lng_max = bounds_src.right
            lat_max = bounds_src.top

        num_bandas = src.count
        if num_bandas >= 3:
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
        else:
            canal = src.read(1).astype(np.float32)
            r = g = b = canal

    # Copernicus exporta True Color já processado em uint16
    # Escala linear preserva as cores exatas sem distorção
    def escalar_uint16_para_uint8(banda: np.ndarray) -> np.ndarray:
        return (banda / 65535.0 * 255).astype(np.uint8)

    imagem_rgb = np.stack([
        escalar_uint16_para_uint8(r),
        escalar_uint16_para_uint8(g),
        escalar_uint16_para_uint8(b),
    ], axis=-1)

    return {
        "imagem_rgb": imagem_rgb,
        "bounds": [[lat_min, lng_min], [lat_max, lng_max]],
    }