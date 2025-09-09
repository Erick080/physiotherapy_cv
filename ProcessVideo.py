import cv2
import os
import sys
import argparse
import mediapipe as mp

def processar_video(video_entrada, video_saida):
    mp_drawing = mp.solutions.drawing_utils
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

            mp_drawing.draw_landmarks(
                frame,
                resultados.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            out.write(frame)

            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # Pressione ESC para sair
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"Vídeo processado salvo em: {video_saida}")

def selecionar_frames_de_video(video):
    if not os.path.isfile(video):
        print(f"Erro: O arquivo '{video}' não foi encontrado.")
        return

    pasta_saida = f"./frames_selecionados/{os.path.basename(video)}"
    
    try:
        os.makedirs(pasta_saida, exist_ok=True)
        print(f"Frames selecionados serão salvos em '{pasta_saida}/'")

    except OSError as e:
        print(f"Erro ao criar o diretório: {e}")
        return

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
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
        ret, frame = cap.read()

        if not ret:
            print("Fim do vídeo.")
            break

        cv2.putText(
            frame,
            f'Frame: {frame_atual_idx}',
            (15, 30), # Posição (x, y) no canto superior esquerdo
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, # Escala da fonte
            (0, 255, 0), # Cor (em BGR - verde)
            2 # Espessura da linha
        )

        cv2.imshow('Seletor de Frames - Pressione "S" para Salvar, "Q" para Sair', frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            nome_arquivo = f"frame_{frames_salvos_count:04d}.jpg"
            caminho_completo = os.path.join(pasta_saida, nome_arquivo)
            
            cv2.imwrite(caminho_completo, frame)
            
            print(f"-> Frame {frame_atual_idx} salvo como '{nome_arquivo}'")
            frames_salvos_count += 1

        elif key == ord('q'):
            print("Saindo do programa.")
            break
                
        frame_atual_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nConcluído! {frames_salvos_count} frames foram salvos na pasta '{pasta_saida}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gera o esqueleto de um vídeo e salva frames selecionados.")
    parser.add_argument('--path', required=True, help='caminho para o vídeo desejado.')
    args = parser.parse_args()
    
    if not os.path.isfile(args.path):
        print(f"Erro: O arquivo '{args.path}' não foi encontrado.")
        sys.exit(1)

    video_entrada = args.path
    nome_arquivo, _ = os.path.splitext(os.path.basename(video_entrada))
    video_saida = f'./videos_processados/{nome_arquivo}.mp4'

    processar_video(video_entrada, video_saida)
    selecionar_frames_de_video(video_saida)