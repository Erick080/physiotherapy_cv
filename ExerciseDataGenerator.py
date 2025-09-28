### Esse script vai fazer algo parecido com o que o detect from image do T2 de ML fazia:
### Ele vai receber um conjunto de imagens (frames) e vai rodar o mediapipe para estimar os keypoints
### Depois vai calcular o angulo de cada tripleto (especificados no MediapipePoseEstimator.py) e salvar em
### um yml na pasta exercises_data, para ser usado futuramente na comparação

import os
import argparse
import yaml
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from GeometryUtils import calcular_angulos_frame
from MediapipePoseEstimation import PRESENCE_THRESHOLD

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Estimador de pose usando MediaPipe")
parser.add_argument('--images_dir', type=str, required=True, help='Diretório contendo imagens de entrada')
parser.add_argument('--model_type', type=str, required=True, choices=['heavy', 'full', 'lite'], help='Tipo de modelo a ser usado')
parser.add_argument('--output_file', type=str, default='poses_output.yml', help='Arquivo de saída YAML')
parser.add_argument('--tipo_exercicio', type=str, required=True, choices=['braco','perna','braco_e_perna'], help='Tipo do exercicio: braco, perna ou braco e perna')
parser.add_argument('--tempo', type=int, required=True, help='Tempo em segundos de alongamento')

args = parser.parse_args()

# Caminhos dos modelos
model_paths = {
  'heavy': os.path.join('models', 'pose_landmarker_heavy.task'),
  'full': os.path.join('models', 'pose_landmarker_full.task'),
  'lite': os.path.join('models', 'pose_landmarker_lite.task')
}
model_path = model_paths[args.model_type]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

pose_outputs = {}
with PoseLandmarker.create_from_options(options) as landmarker:
    filenames = sorted(os.listdir(args.images_dir))
    for idx, filename in enumerate(filenames):
        image_path = os.path.join(args.images_dir, filename)
        mp_image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(mp_image)

        frame_key = f'frame_{idx}'
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            landmarks_filtrados = [
                lm if getattr(lm, "presence", 1.0) >= PRESENCE_THRESHOLD else None
                for lm in landmarks
            ]            
            pose_outputs[frame_key] = calcular_angulos_frame(landmarks_filtrados)
        else:
            pose_outputs[frame_key] = {}

#print(pose_outputs)

yaml_dict = {
    'tipo_exercicio' : args.tipo_exercicio,
    'tempo_alongamento' : args.tempo,
    'frames' : pose_outputs
}

with open(args.output_file, 'w') as f:
    yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=False)
