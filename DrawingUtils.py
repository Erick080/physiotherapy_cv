import mediapipe as mp
import cv2
import os

from GeometryUtils import TRIPLETOS

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_skeleton(frame, landmarks_filtrados, tripletos_errados):
    for a_idx, b_idx, c_idx in TRIPLETOS:
        pontos = [landmarks_filtrados[a_idx], landmarks_filtrados[b_idx], landmarks_filtrados[c_idx]]
        if None not in pontos:
            x1, y1 = int(pontos[0].x * frame.shape[1]), int(pontos[0].y * frame.shape[0])
            x2, y2 = int(pontos[1].x * frame.shape[1]), int(pontos[1].y * frame.shape[0])
            x3, y3 = int(pontos[2].x * frame.shape[1]), int(pontos[2].y * frame.shape[0])

            if (a_idx, b_idx, c_idx) in tripletos_errados:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
                cv2.line(frame, (x3, y3), (x2, y2), (0, 0, 255), 6)
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
                cv2.line(frame, (x3, y3), (x2, y2), (0, 255, 0), 6)

    return frame
    

def draw_stats(frame, pose_index, num_poses, reps, timer_alongamento):
    # Define tamanho e posição do retângulo de fundo
    x, y = 0, 20
    largura = 350
    altura = 150
    # Desenha retângulo preto semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + largura, y + altura), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Mostra número de poses detectadas e quantas faltam para acabar o exercicio
    cv2.putText(frame, f"Pose {pose_index}/{num_poses}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Mostra numero de repetições
    cv2.putText(frame, f"Reps: {reps}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Timer de alongamento
    cv2.putText(frame, f"Hold Time: {timer_alongamento:.2f}s", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame


def load_ref_img(exercicio_imgs_dir, exercicio_imgs_filenames, pose_index):
    ref_img_path = os.path.join(exercicio_imgs_dir, exercicio_imgs_filenames[pose_index])
    ref_img_loaded = cv2.imread(ref_img_path)
    ref_img_loaded = cv2.flip(ref_img_loaded, 1)
    return ref_img_loaded
        






