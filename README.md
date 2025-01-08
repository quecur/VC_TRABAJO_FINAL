# VC_TRABAJO_FINAL
# VIRTUALIZACIÓN DE UN TABLERO DE AJEDREZ
# CARLOS TOMÁS QUEVEDO OLIVARES

![image](https://github.com/user-attachments/assets/03965ae1-802a-4d24-8820-0c8b30e0830a)

# ENLACE VIDEO DEMOSTRACIÓN
https://drive.google.com/file/d/18292oUll6Ew-xjoHFV9fev_F0rqiPTwZ/view?usp=sharing

# INTRODUCCIÓN

Actualmente y desde hace mucho tiempo la tecnología empleada para reproducir partidas de ajedrez a nivel competitivo consiste en la utilización de tableros electrónicos de alto coste con sensores físicos para registrar el movimiento y la posición de las piezas. Es por ello por lo que, resulta interesante realizar una aproximación alternativa mediante el uso de visión por computador para este propósito.

# OBJETIVO

Para el desarrollo de esta práctica me he centrado fundamentalmente en la detección y virtualización de una posición correspondiente a una imagen estática en un tablero de ajedrez, para ello decidí enfocarme necesariamente en cuatro aspectos principales. 
-	Homografía del tablero: establecimiento de una relación entre el tablero que se quiere virtualizar y su equivalencia respecto a un tablero normalizado desde una perspectiva cenital.
-	Detección de las piezas: detección y clasificación de las piezas que se encuentran en el tablero.
-	Posición de la piezas en el tablero: una vez realizados los dos puntos anteriores conocer la casilla en la que se encuentra cada pieza.
-	Virtualización: en último lugar virtualización de la posición

# TECNOLOGÍAS UTILIZADAS

Para el desarrollo de está practica he utilizado YOLO para el entramiento de un modelo capaz de detectar y clasificar las diferentes piezas de ajedrez además de, las siguientes librerías de Python:
-	cv2 (OpenCV): Proporciona herramientas para procesamiento de imágenes y visión por computadora, como carga, manipulación y análisis de imágenes.
-	YOLO (Ultralytics): Necesaria para manipular un modelo entrenado para la detección de piezas de ajedrez basado en la red neuronal YOLO.
-	matplotlib.pyplot: Permite crear gráficos y visualizar datos, como imágenes, histogramas o gráficos de dispersión.
-	numpy (np): Proporciona estructuras de datos y operaciones avanzadas para cálculos matemáticos y manipulación de matrices.
-	skimage.transform: Ofrece herramientas para transformar imágenes, como redimensionar, rotar, escalar y ajustar perspectivas(homografía).
-	pygame: Librería para el desarrollo de videojuegos en Python, proporcionando herramientas para gráficos 2D, sonido, manejo de eventos y control de usuarios.
-	chess: Biblioteca para crear y manipular juegos de ajedrez, permitiendo implementar movimientos legales, análisis de posiciones, generación de tableros y soporte para formatos estándar como PGN y FEN.

![image](https://github.com/user-attachments/assets/591b214a-85e2-4d7b-a03d-b8ae72bcd32c)

![image](https://github.com/user-attachments/assets/3bc87736-ea8a-490e-9b14-b07937c309fb)

# DESARROLLO

# Homografía 

En primer lugar generamos un tablero de ajedrez normalizado haciendo uso de las herramientas de openCV. 
Decidí generarlo de esta forma en lugar de sacarlo de internet puesto que, así conozco previamente las dimensiones del tablero y por extensión las coordenadas de cada casilla, esto resultara muy útil en el futuro tanto para realizar la homografía, como para conocer la posición de las piezas.

![image](https://github.com/user-attachments/assets/d13719ed-7026-40aa-9b0d-c9f5dc81326b)

En segundo lugar, obtenemos las coordenadas correspondientes a las esquinas de cada tablero, las del tablero que acabamos de crear ya las conocemos previamente por lo que simplemente las guardamos, por otra parte las coordenadas del tablero que queremos detectar las obtenemos de forma manual clicando en las correspondientes esquinas del tablero. El orden para hacerlo(muy importante) seria(desde una perspectiva de las blancas), esquina inferior izquierda, inferior derecha, superior izquierda, superior derecha. 

![image](https://github.com/user-attachments/assets/4e23c486-de8a-49f3-aca4-e58acc4d3a87)

![image](https://github.com/user-attachments/assets/c68988ae-1ce5-4c76-919d-d5a716dbb5df)

Así quedaría la imagen generada de la vista del tablero desde un plano cenital.

![image](https://github.com/user-attachments/assets/0fb24d38-2525-4071-a6f4-0d3ba55a1a59)

Con la transformación realizada, podemos movernos por el tablero objeto de estudio y comprobar su posición correspondiente en el tablero normalizado.

![image](https://github.com/user-attachments/assets/9326c9bd-f484-4eaa-a121-e0d14a4cec23)

# Detección y Posición

Para realizar la detección de las piezas de ajedrez en el tablero debemos entrenar un modelo para ello. Al igual que en ocasiones anteriores y aprovechando la misma configuración,  he hecho uso de YOLO, utilizando la GPU para el entrenamiento.
El dataset correspondiente al entrenamiento del modelo fue extraído de la página roboflow y se puede obtener en el siguiente enlace.
https://universe.roboflow.com/joseph-nelson/chess-pieces-new/dataset/24

![image](https://github.com/user-attachments/assets/03b6b29b-b5fe-4e00-b209-f4a14c0314fd)

![image](https://github.com/user-attachments/assets/55654f0b-948b-4c85-9c47-1df5c0aa6e02)

Desde las primeras épocas los resultado apuntan muy bien, la detección de las cajas parece ser muy precisa según la variable mAP50, por otra parte, box_loss, cls_lss y dfl_lss que hacen referencia a la posición de las cajas que se detectan y a la clasificación del objeto en su interior, bajan de forma continua según avanza el entramiento. 
El entrenamiento fue de 80 épocas pues hasta entonces los resultados fueron mejorando. A la hora de probar el modelo este me proporciono detecciones bastante buenas por lo que decidí escogerlo para la práctica.
Las siguientes gráficas muestran la mejora continua de los indicadores del entrenamiento comentados anteriormente para las 80 épocas.

![image](https://github.com/user-attachments/assets/09c54678-c791-41d2-8da6-da100dc1288d)

Una vez listo el modelo, procedemos a realizar la detección sobre la imagen que queremos estudiar, las imágenes seleccionadas para hacer las pruebas fueron extraídas de la carpeta test del dataset. Evidentemente, la imagen sobre la que realizamos la homografía y sobre la que realizamos la detección debe coincidir.
Inicializamos el modelo y creamos una matriz vacía de 8x8 donde guardaremos la pieza detectada y su posición en el tablero indicada por el índice de la matriz donde guardaremos la pieza.

![image](https://github.com/user-attachments/assets/ae1efdf3-2d6a-4b96-ab18-8f596c8d6bdc)

Realizamos lo descrito anteriormente, es decir, guardamos la pieza en sus respectiva posición en la matriz, pasando las coordenadas de la caja detectada en la imagen original por una función que la transforma en su posición correspondiente en el tablero normalizado. Esto es posible gracias a lo realizado en el primer punto de la práctica. Dependiendo de las coordenadas obtenidas gracias a esa función se tratará de una casilla u otra y podremos guardarla en la matriz vacía en su lugar correspondiente.

![image](https://github.com/user-attachments/assets/38fbf61e-62f1-46eb-97ff-d13d01af9c64)

Finalmente como paso previo al último apartado, aprovechamos y transformamos la matriz de posiciones obtenida en una línea de texto de formato fen. El formato fen es un tipo de anotación ajedrecística legible para librerías de programación como chess en Python, con la que se generara la posición resultado de la detección.

![image](https://github.com/user-attachments/assets/99d277ae-6f31-4b43-8ddb-a77ab59f5f98)

Esta sería la imagen que refleja la salida de las piezas detectadas, la matriz de posición y la línea de texto en formato fen.

![image](https://github.com/user-attachments/assets/f1676102-39d2-4726-ae13-463532895210)

La detección para una imagen cualquiera perteneciente al conjunto test vemos que se realiza con una precisión total.

![image](https://github.com/user-attachments/assets/2a6b0a11-bc0d-47d4-b9c6-f1e74faf9695)

Posición de las piezas detectadas en el tablero normalizado.

![image](https://github.com/user-attachments/assets/8a769f7c-f37f-4d57-9bad-d84a1bdedcb9)

# Virtualización 

Por último con todo listo y la línea de texto en formato fen preparada, generamos la posición del tablero con la librería chess y, con ayuda de la librería pygame creamos el tablero de forma gráfica. Para las piezas del tablero fue necesario extraer imágenes en formato png de internet.
Inicialización de pygame y establecimiento de la posición.

![image](https://github.com/user-attachments/assets/0fa0d74f-f326-42bc-8f8c-193c6290531b)

Función para crear el tablero.

![image](https://github.com/user-attachments/assets/344f4300-1e5d-4893-ae8a-dfc77e77801e)

Pngs correspondientes a las piezas del tablero sacados de internet.

![image](https://github.com/user-attachments/assets/17fa0b25-05e2-4ccf-9467-47e2480d2e32)

![image](https://github.com/user-attachments/assets/2764478a-1335-4d69-a2be-29c39fc39e74)

Posición final virtualizada en un tablero generado con pygames.

![image](https://github.com/user-attachments/assets/e7db6772-1a26-480e-8ac3-2d2f371bed8a)

El resultado es un éxito total, la posición virtualizada corresponde exactamente a la de su origen en el tablero físico. Esto se puede apreciar más fácilmente comparándola con la imagen orinal del tablero rotada.

![image](https://github.com/user-attachments/assets/072e3ccf-57e0-4f24-8be1-ceb906a14baa)

# CONCLUSIÓN Y PROPUESTAS DE AMPLIACIÓN 

Los objetivos de la práctica han sido cumplidos de forma muy exitosa, si se prueba el programa con otras imágenes pertenecientes al directorio test del conjunto de datos funciona muy bien, no obstante, existe un buen margen de mejora detallado en las siguientes propuestas de ampliación.
-	Detección automática de esquinas: para la detección del tablero se podría decir que hago un poco de trampa puesto que, realizo de forma manual la especificación de las coordenadas de este, lo ideal es que este proceso fuese automático y mas teniendo el cuenta el siguiente punto.
-	Detección en video: implementación de la práctica para la detección en video de una partida de ajedrez, idealmente en tiempo real.
-	Mejora del dataset y detección: con el conjunto de datos que se ha entrenado el modelo, este funciona muy bien para imágenes del tablero desde esas perspectivas, ese tipo de tablero, esa escala y esas piezas concretas,  sin embargo, para imágenes en las que se alteran un poco estas variables tiene dificultades, sería muy interesante una mayor generalización del modelo mediante el uso de técnicas de aumentación de los datos y al mismo tiempo ampliación manual del dataset.
-	Mejora de la interfaz del tablero virtualizado: también sería muy interesante mejorar la interfaz gráfica del tablero virtual, ahora mismo solo muestra la posición y no se puede interactuar con él. Estaría bien que se pudiese modificar la posición haciendo jugadas, que se mostrase quien va ganando y jugadas candidatas.
-	Integración en una aplicación: si se consigue todo lo anterior se podría integrar en una aplicación.




















