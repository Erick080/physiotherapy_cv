import cv2
import time
import argparse
import os
import yaml
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

from GeometryUtils import calcular_angulos_frame, comparar_angulos

PRESENCE_THRESHOLD = 0.5   # Limite de presença para considerar um landmark válido

LARGURA_JANELA = 1890
ALTURA_JANELA = 1020

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
while True:
    try:
        escolha = int(input("\nDigite o número do exercício desejado: "))
        if 1 <= escolha <= len(exercicios):
            exercicio_selecionado = exercicios[escolha-1]
            dados_exercicio_selecionado = exercicios_dados[escolha-1]
            print(f"Exercício selecionado: {exercicio_selecionado}")
            break
        else:
            print("Número inválido, tente novamente.")
    except ValueError:
        print("Entrada inválida, digite um número.")

tipo_exercicio = dados_exercicio_selecionado.get('tipo_exercicio')
angulos_ref = dados_exercicio_selecionado.get('frames', {})

# Configurações do PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Utilitário de desenho
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cria o detector
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=1
)

# Inicia a webcam e configuracoes da janela do opencv
window_name = "MediaPipe Pose (Tasks API)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, LARGURA_JANELA, ALTURA_JANELA)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

# Configuracao de feedback sonoro
pygame.mixer.init()
success_sound = pygame.mixer.Sound('success_bell.mp3')

with PoseLandmarker.create_from_options(options) as landmarker:
    pose_index = 0 # Index da pose atual sendo usada na comparacao
    reps = 0 # Contador de repetições
    while True:
        start_time = time.time()
        
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

            # Verificar se é melhor desenhar as linhas antes ou depois do cálculo de corretude
            mp_drawing.draw_landmarks(
                frame,
                landmark_list,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            angulos_detect_frame = calcular_angulos_frame(landmarks_filtrados)
            angulos_ref_frame = angulos_ref.get(f'frame_{pose_index}', {})
            if comparar_angulos(angulos_detect_frame, angulos_ref_frame, tipo_exercicio, DEBUG):
                pose_index += 1
                success_sound.play()
                if pose_index >= len(angulos_ref):
                    pose_index = 0
                    reps += 1

        # Mostra número de poses detectadas e quantas faltam para acabar o exercicio
        num_poses = len(angulos_ref)
        cv2.putText(frame, f"Pose {pose_index}/{num_poses}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Mostra numero de repetições
        cv2.putText(frame, f"Reps: {reps}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # FPS
        if DEBUG:
            fps = 1 / (time.time() - start_time + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostra a imagem
        frame_redimensionado = cv2.resize(frame, (LARGURA_JANELA, ALTURA_JANELA), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, frame_redimensionado)
        
        # Verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
