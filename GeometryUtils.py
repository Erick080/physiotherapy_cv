### Calculos relacionados ao angulo entre vetores, calculo de erro quadratico, etc
import math
import numpy as np

TRIPLETOS = [
    (11, 13, 15),    # OMBRO E BRACO ESQUERDO
    (12, 14, 16),    # OMBRO E BRACO DIREITO
    (11, 23, 25),    # TORSO E CINTURA ESQUERDA
    (12, 24, 26),    # TORSO E CINTURA
    (23, 25, 27),    # PERNA ESQUERDA 
    (24, 26, 28),    # PERNA DIREITA        
]

def calcular_angulo_2d(a, b, c):    # calcula o angulo de um tripleto (angulo entre AB e CB)
    # calcula vetores ab e cb
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    # calcula produto escalar: u . v = x1 * x2 + y1 * y2
    produto_escalar = ab[0]*cb[0] + ab[1]*cb[1]
    # norma (magnitude) dos vetores
    norma_ab = math.hypot(*ab)
    norma_cb = math.hypot(*cb)
    if norma_ab == 0 or norma_cb == 0:
        return None
    # cos(x) = (u . v) / (|u| * |v|)      
    cos_angulo = produto_escalar / (norma_ab * norma_cb)
    cos_angulo = max(-1.0, min(1.0, cos_angulo)) # evita erros de arredondamento onde o cos pode ser maior que 1 ou menor que -1
    angulo_rad = math.acos(cos_angulo)
    return math.degrees(angulo_rad)

def calcular_angulos_frame(landmarks):  # calcula o angulo de todos tripletos em um frame
    angulos = {}
    for a_idx, b_idx, c_idx in TRIPLETOS:
        p_a = landmarks[a_idx]
        p_b = landmarks[b_idx]
        p_c = landmarks[c_idx]
        angulo = calcular_angulo_2d(p_a, p_b, p_c)
        if angulo is not None:
            angulos[f"{a_idx}-{b_idx}-{c_idx}"] = angulo
    return angulos

def comparar_angulos(ang_atual, ang_salvo, threshold=20):
    valores_comparados = []
    for chave, angulo_salvo in ang_salvo.items():
        if chave in ang_atual and isinstance(angulo_salvo, (int, float)):
            # calcula erro quadratico entre os angulos
            valores_comparados.append((ang_atual[chave] - angulo_salvo) ** 2)
    if not valores_comparados:
        return False
    # Root Mean Squared Error -> raiz da media dos erros quadraticos
    rmse = np.sqrt(np.mean(valores_comparados))
    print(f"RMSE: {rmse:.2f} (Threshold: {threshold})")
    return rmse < threshold