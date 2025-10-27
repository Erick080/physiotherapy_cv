import cv2
import os
import sys
import argparse
import mediapipe as mp
import subprocess
from DrawingUtils import draw_skeleton

def processar_video(video_entrada, video_saida):
    mp_pose = mp.solutions.pose

    output_dir = os.path.dirname(video_saida)
    os.makedirs(output_dir, exist_ok=True)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        cap = cv2.VideoCapture(video_entrada)

        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_saida, fourcc, fps, (largura, altura))

        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso:
                print("Fim do vídeo ou erro na leitura.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resultados = pose.process(frame_rgb)

            if resultados.pose_landmarks:
                landmarks_filtrados = list(resultados.pose_landmarks.landmark)
                frame = draw_skeleton(frame, landmarks_filtrados, [])

            out.write(frame)

            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # Pressione ESC para sair
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"Vídeo processado salvo em: {video_saida}")

def selecionar_frames_de_video(video_original, video_processado, nome_exercicio):
    if not os.path.isfile(video_original) or not os.path.isfile(video_processado):
        print(f"Erro: Um dos arquivos de video não foi encontrado.")
        return

    pasta_saida = f"./exercises_input/{nome_exercicio}"
    
    try:
        os.makedirs(pasta_saida, exist_ok=True)
        print(f"Frames selecionados serão salvos em '{pasta_saida}/'")
    except OSError as e:
        print(f"Erro ao criar o diretório: {e}")
        return
    
    cap_original = cv2.VideoCapture(video_original)
    cap_processado = cv2.VideoCapture(video_processado)

    if not cap_original.isOpened() or not cap_processado.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    frame_atual_idx = 0
    frames_salvos_count = 0
    
    print("\n--- Controles ---")
    print("  [S]   -> Salvar o frame atual")
    print("  [Q]   -> Sair do programa")
    print("  Qualquer outra tecla -> Avançar para o próximo frame")
    print("-----------------\n")

    while True:
        ret_original, frame_original = cap_original.read()
        ret_processado, frame_processado = cap_processado.read()

        if not ret_original or not ret_processado:
            print("Fim do vídeo.")
            break

        cv2.putText(
            frame_processado,
            f'Frame: {frame_atual_idx}',
            (15, 30), # Posição (x, y) no canto superior esquerdo
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, # Escala da fonte
            (0, 255, 0), # Cor (em BGR - verde)
            2 # Espessura da linha
        )

        cv2.imshow('Seletor de Frames - Pressione "S" para Salvar, "Q" para Sair', frame_processado)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            nome_arquivo = f"frame_{frames_salvos_count:04d}.jpg"
            caminho_completo = os.path.join(pasta_saida, nome_arquivo)
            
            cv2.imwrite(caminho_completo, frame_original)
            
            print(f"-> Frame {frame_atual_idx} salvo como '{nome_arquivo}'")
            frames_salvos_count += 1

        elif key == ord('q'):
            print("Saindo do programa.")
            break
                
        frame_atual_idx += 1

    cap_original.release()
    cap_processado.release()
    cv2.destroyAllWindows()
    print(f"\nConcluído! {frames_salvos_count} frames foram salvos na pasta '{pasta_saida}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gera o esqueleto de um vídeo e salva frames selecionados.")
    parser.add_argument('--path', required=True, help='caminho para o vídeo desejado.')
    parser.add_argument('--exercise_name', required=True, help='Nome do exercício a ser registrado.')
    parser.add_argument('--exercise_type', required=True, choices=['braco','perna','braco_e_perna'], help='Tipo do exercício: braco, perna ou braco e perna')
    parser.add_argument('--hold_time', type=int, required=True, help='Tempo em segundos que a pose deve ser mantida em segundos.')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.path):
        print(f"Erro: O arquivo '{args.path}' não foi encontrado.")
        sys.exit(1)

    video_entrada = args.path
    nome_exercicio = args.exercise_name
    video_saida = f'./videos_processados/{nome_exercicio}.mp4'

    processar_video(video_entrada, video_saida)
    selecionar_frames_de_video(video_entrada, video_saida, nome_exercicio)

    # Reaproveita o codigo para calcular os angulos dos frames de um diretorio
    comando = [
        'python', './ExerciseDataGenerator.py',
        '--images_dir', f'./exercises_input/{nome_exercicio}',
        '--model_type', 'heavy',
        '--output_file', f'./exercises_output/{nome_exercicio}.yaml',
        '--tipo_exercicio', args.exercise_type,
        '--tempo', str(args.hold_time)
    ]

    try:
        result = subprocess.run(comando, check=True, capture_output=True, text=True)
        print("Comando executado com sucesso!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando: {e}")
        print(f"Saída de erro: {e.stderr}")