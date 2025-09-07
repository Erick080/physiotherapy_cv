### Calculos relacionados ao angulo entre vetores, calculo de erro quadratico, etc
import math
import numpy as np

POSE_ERROR_THRESHOLD = 15  # RMSE mínimo para detectar uma pose como correta 

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

def calcular_angulos_frame(landmarks, debug=False):  # calcula o angulo de todos tripletos em um frame
    angulos = {}
    for a_idx, b_idx, c_idx in TRIPLETOS:
        # Verifica se algum dos landmarks é None (presença baixa)
        if (landmarks[a_idx] is None or
            landmarks[b_idx] is None or
            landmarks[c_idx] is None):
            if debug:
                print(f'\nNao conseguiu detectar a posicao do tripleto ({a_idx}-{b_idx}-{c_idx})')
            continue

        p_a = landmarks[a_idx]
        p_b = landmarks[b_idx]
        p_c = landmarks[c_idx]
        angulo = calcular_angulo_2d(p_a, p_b, p_c)
        angulos[f"{a_idx}-{b_idx}-{c_idx}"] = angulo
    return angulos

# Compara os angulos de todos tripletos no frame, retorna o booleano indicando se 
# a pose esta correta e uma lista com os tripletos que estao errados
def comparar_angulos(angulos_detec, angulos_salvos, tipo_exercicio, debug=False, segurando_alongamento=False):
    threshold_ajustado = POSE_ERROR_THRESHOLD * 4 if segurando_alongamento else POSE_ERROR_THRESHOLD 

    valores_comparados = []
    tripletos_errados = []
    for chave, angulo_salvo in angulos_salvos.items():
        if chave in angulos_detec:
            # calcula erro quadratico entre os angulos
            erro_quadratico = (angulos_detec[chave] - angulo_salvo) ** 2
            valores_comparados.append(erro_quadratico)
            if erro_quadratico > (POSE_ERROR_THRESHOLD ** 2):
                tripletos_errados.append(tuple(map(int, chave.split('-'))))                

        elif tipo_exercicio == 'braco' and chave in ['11-13-15', '12-14-16']:
            if debug:
                print("Nao detectou braco")
            return False, []
        
        elif tipo_exercicio == 'perna' and chave in ['23-25-27', '24-26-28']:
            if debug:
                print("Nao detectou pernas")
            return False, []

    if not valores_comparados: # se nao detectou nada retorna falso
        return False
    # Root Mean Squared Error -> raiz da media dos erros quadraticos
    rmse = np.sqrt(np.mean(valores_comparados))
    if debug:
        print(f"RMSE: {rmse:.2f} (Threshold: {POSE_ERROR_THRESHOLD})")
    return rmse < threshold_ajustado, tripletos_errados