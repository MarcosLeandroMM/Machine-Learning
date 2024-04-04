"""
Manipulação de Dados Geoespaciais:

Trabalhe com dados geoespaciais usando bibliotecas como GeoPandas, fazendo operações de plotagem e análise.

"""

import geopandas as gpd
import matplotlib.pyplot as plt

# Carregando um shapefile de exemplo (pontos de cidades no Brasil)
shapefile_path = "cidades_br.shp"
gdf_cidades = gpd.read_file(shapefile_path)

# Exibindo as primeiras linhas do GeoDataFrame
print(gdf_cidades.head())

# Plotando o mapa das cidades
gdf_cidades.plot()
plt.title('Mapa das Cidades no Brasil')
plt.show()

"""
gpd.read_file(shapefile_path) é usado para ler um shapefile que contém informações geoespaciais. Você pode encontrar dados geoespaciais em formatos como Shapefile, GeoJSON, entre outros.

gdf_cidades.plot() é utilizado para plotar o mapa das cidades.

"""
