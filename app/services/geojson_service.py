from shapely.geometry import Polygon
from pyproj import Transformer

def pixels_para_latlng(poligono: list, bounds: list, largura: int, altura: int) -> list:
    """Converte polígono em pixels para coordenadas lat/lng usando bounds da sessão."""
    [[lat_min, lng_min], [lat_max, lng_max]] = bounds
    coordenadas = []
    for x, y in poligono:
        lng = lng_min + (x / largura) * (lng_max - lng_min)
        lat = lat_max - (y / altura) * (lat_max - lat_min)
        coordenadas.append([lng, lat])  # GeoJSON usa [lng, lat]
    return coordenadas


def calcular_area_hectares(coordenadas_lnglat: list) -> float:
    """
    Calcula área em hectares usando aproximação plana.
    Converte lat/lng para metros via projeção local (UTM aproximado),
    calcula área com shapely e converte m² → hectares.
    """
    # Pega centroide para definir zona de projeção local
    lngs = [c[0] for c in coordenadas_lnglat]
    lats = [c[1] for c in coordenadas_lnglat]
    lng_centro = sum(lngs) / len(lngs)
    lat_centro = sum(lats) / len(lats)

    # Projeção azimutal equidistante centrada no polígono — boa aproximação plana local
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat_centro} +lon_0={lng_centro} +units=m",
        always_xy=True
    )

    coords_metros = [transformer.transform(lng, lat) for lng, lat in coordenadas_lnglat]
    area_m2 = Polygon(coords_metros).area
    return round(area_m2 / 10_000, 4)  # m² → hectares


def montar_geojson(talhoes: list, bounds: list, largura: int, altura: int) -> dict:
    """
    Monta FeatureCollection GeoJSON a partir dos talhões da sessão.
    Se bounds existir, converte pixels → lat/lng.
    Se não, exporta em pixels como coordenadas brutas.
    """
    features = []

    for talhao in talhoes:
        if bounds:
            coordenadas = pixels_para_latlng(talhao["poligono"], bounds, largura, altura)
            area_ha = calcular_area_hectares(coordenadas)
        else:
            # Sem georeferência: exporta pixels diretamente
            coordenadas = [[x, y] for x, y in talhao["poligono"]]
            area_ha = None

        # GeoJSON Polygon exige que o primeiro e último ponto sejam iguais
        if coordenadas[0] != coordenadas[-1]:
            coordenadas.append(coordenadas[0])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordenadas]
            },
            "properties": {
                "id": talhao["id"],
                "area_pixels": talhao["area_pixels"],
                "area_hectares": area_ha,
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }