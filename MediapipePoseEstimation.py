import cv2
import time
import argparse
import os
import yaml
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

from GeometryUtils import calcular_angulos_frame, comparar_angulos
from DrawingUtils import draw_skeleton, draw_stats, load_ref_img

PRESENCE_THRESHOLD = 0.5   # Limite de presença para considerar um landmark válido

LARGURA_JANELA = 1920
ALTURA_JANELA = 1080

largura_esquerda = int(LARGURA_JANELA * 2 / 3)
largura_direita = LARGURA_JANELA - largura_esquerda  # 1/3 da largura


DEBUG = False

# Parser de argumentos
parser = argparse.ArgumentParser(description="Estimativa de pose com MediaPipe Tasks API")
parser.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                    help="Escolha o modelo: lite, full ou heavy (padrão: full)")
args = parser.parse_args()

# Mapeamento do modelo
model_paths = {
    "lite": "models/pose_landmarker_lite.task",
    "full": "models/pose_landmarker_full.task",
    "heavy": "models/pose_landmarker_heavy.task"
}
model_path = model_paths[args.model]

# Carregamento de exercícios cadastrados
exercicios = [f for f in os.listdir("exercises_output")]
if not exercicios:
    print("ERRO: Nenhum exercício cadastrado")
    exit()

exercicios_dados = [] # array que contem os angulos de todos exercicios cadastrados
for nome_arquivo in exercicios:
    caminho = os.path.join("exercises_output", nome_arquivo)
    with open(caminho, "r") as f:
        dados = yaml.safe_load(f)
        exercicios_dados.append(dados)

# Menu de escolha de exercício
for idx, nome in enumerate(exercicios):
    print(f"{idx+1}. {nome}")

dados_exercicio_selecionado = {}
exercicio_imgs = []

while True:
    try:
        escolha = int(input("\nDigite o número do exercício desejado: "))
        if 1 <= escolha <= len(exercicios):
            # Add angulos do exercicio em um dicionario (chave = frame valor = angulos)
            dados_exercicio_selecionado = exercicios_dados[escolha-1]

            # Pega todas imagens do exercicio
            exercicio_selecionado = exercicios[escolha-1]
            exercicio_nome = exercicio_selecionado.replace('.yaml', '')
            exercicio_img_dir = os.path.join("exercises_input", exercicio_nome)
            exercicio_imgs = sorted([
                f for f in os.listdir(exercicio_img_dir)
            ])

            print(f"Exercício selecionado: {exercicio_nome}")
            break
        else:
            print("Número inválido, tente novamente.")
    except ValueError:
        print("Entrada inválida, digite um número.")

tipo_exercicio = dados_exercicio_selecionado.get('tipo_exercicio')
tempo_alongamento = dados_exercicio_selecionado.get('tempo_alongamento') # segundos
angulos_ref = dados_exercicio_selecionado.get('frames', {})

# Configurações do PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Cria o detector
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=1
)

# Inicia a webcam e configuracoes da janela do opencv
window_name = "Computer Vision Physiotherapy"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, LARGURA_JANELA, ALTURA_JANELA)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

# Configuracao de feedback sonoro
pygame.mixer.init()
success_sound = pygame.mixer.Sound('success_bell.mp3')

inicio_alongamento = 0
timer_alongamento = 0

with PoseLandmarker.create_from_options(options) as landmarker:

    pose_index = 0 # Index da pose atual sendo usada na comparacao
    ref_img_loaded = load_ref_img(exercicio_img_dir, exercicio_imgs, pose_index)
    reps = 0 # Contador de repetições
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(time.time() * 1000)

        ### DETECCAO SINCRONA:
        result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            landmarks_filtrados = []
            for lm in landmarks:
                landmark = landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z,
                    visibility=lm.visibility, presence=lm.presence
                )
                landmark_list.landmark.append(landmark)

                if lm.presence >= PRESENCE_THRESHOLD:
                    landmarks_filtrados.append(lm)
                else:
                    landmarks_filtrados.append(None)
            
            angulos_detect_frame = calcular_angulos_frame(landmarks_filtrados)
            angulos_ref_frame = angulos_ref.get(f'frame_{pose_index}', {})

            pose_correta, tripletos_errados = comparar_angulos(
                angulos_detect_frame, angulos_ref_frame, tipo_exercicio, DEBUG,
                (tempo_alongamento > 0 and inicio_alongamento > 0) # indica se esta segurando o alongamento
            )

            draw_skeleton(frame, landmark_list, landmarks_filtrados, tripletos_errados)

            if pose_correta:
                if inicio_alongamento == 0:
                    inicio_alongamento = time.time()
                else:
                    timer_alongamento = time.time() - inicio_alongamento
                    if timer_alongamento >= tempo_alongamento:
                        pose_index += 1
                        success_sound.play()
                        timer_alongamento = 0
                        if pose_index >= len(angulos_ref):
                            pose_index = 0
                            reps += 1

                        ref_img_loaded = load_ref_img(exercicio_img_dir, exercicio_imgs, pose_index)
            else:
                inicio_alongamento = 0
                timer_alongamento = 0

        frame = cv2.flip(frame, 1) # inverte no eixo x por causa do espelhamento da camera

        num_poses = len(angulos_ref)
        draw_stats(frame, pose_index, num_poses, reps, timer_alongamento)

        ref_img_resized = cv2.resize(ref_img_loaded, (largura_direita , ALTURA_JANELA), interpolation=cv2.INTER_LINEAR)
        
        # Junta img de exec com de ref lado a lado
        frame_redimensionado = cv2.resize(frame, (largura_esquerda, ALTURA_JANELA), interpolation=cv2.INTER_LINEAR)
        frame_display = np.hstack((frame_redimensionado, ref_img_resized))

        # Mostra a imagem
        cv2.imshow(window_name, frame_display)
        # Verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
