<h1> Physiotherapy Computer Vision Application </h1>
Uma aplicação de monitoramento de execução de exercícios fisioterapêuticos utilizando visão computacional. O modelo de estimação de pose MediaPipe Pose Landmarker é utilizado para a detecção do movimento corporal e o OpenCV como interface gráfica para feedback de execução.
<img width="1684" height="941" alt="bremm_alongamentoperna" src="https://github.com/user-attachments/assets/b27cfcad-aed0-48fa-89c0-5254c4e6ecfe" />

<h2>Instruções de Execução: </h2>

* Inicie o ambiente virtual do python: `python3 -m venv .venv`

OBS: Necessário python 3.12

* Ative o ambiente virtual:

    Windows: `./.venv/Scripts/Activate`

    Linux: `source .venv/bin/activate`

* Instale as dependências:
    
    Windows: `pip install -r windows_requirements.txt`

    Linux: `pip install -r linux_requirements.txt`

    Caso ocorram problemas na instalação, especifique manualmente ao pip os pacotes a serem instalados:

    `pip install mediapipe opencv-python` 

* Instale o modelo do pose landmarker que preferir aqui e coloque ele na pasta models, sem alterar o nome do arquivo.

* Para executar os programas:

    Monitoramento de exercício: `python MediapipePoseEstimation.py --model x`

    Onde x = modelo do pose landmarker escolhido: `lite`, `full` ou `heavy`

	Cadastro de exercício: `python ProcessVideo.py --path ~/path/to/video.mov --exercise_name example_name --hold_time 5 --exercise_type x`
	
	Onde x = tipo de exercício cadastrado: `braco`, `perna`, `braco_e_perna`
