BSS exhaustivo por tamaños sobre bloques completos; CV exclusivamente en Train.

La parada temprana no garantiza un óptimo global entre todos los tamaños.
La CV evalúa el CSV ya transformado: no vuelve a aprender el preprocesamiento
por fold y no debe interpretarse como validación independiente de ese proceso.


min_mejora_pct se fija antes de buscar. No se preseleccionan candidatas. El costo máximo supone llegar a 12; la parada por mejora puede reducirlo y su alcance no se conoce de antemano.
