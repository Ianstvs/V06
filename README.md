# Documentación Técnica Completa: Framework de Aprendizaje Profundo para Grafos Moleculares (GCN)

Este manual proporciona una descripción exhaustiva del funcionamiento, la arquitectura y la configuración del framework diseñado para predecir propiedades fisicoquímicas continuas (como la solubilidad `SOL`) a partir de representaciones de texto **SMILES**. El sistema combina **PyTorch**, **PyTorch Geometric**, **RDKit** y **Optuna** para ofrecer una tubería (*pipeline*) automatizada que abarca desde la curación estadística de datos hasta la optimización bayesiana de la arquitectura de la red y sus hiperparámetros.

---

## 1. Arquitectura del Sistema y Descripción de Módulos

El framework está diseñado bajo un enfoque modular y desacoplado, donde cada script asume una responsabilidad única dentro del flujo de trabajo:

```
├── run.py                 # Punto de entrada principal y configuración del entorno físico.
└── utils/
    ├── Manager.py         # Orquestador global de experimentos y optimización bayesiana (Optuna).
    ├── DataHandler.py     # Extractor de datos, generador de grafos moleculares y DataLoaders optimizados.
    ├── Normalizer.py      # Filtro químico y encargado de la transformación de potencia estadística.
    ├── Itinerary.py       # Administrador del ciclo de entrenamiento, Schedulers y Early Stopping.
    ├── Engine.py          # Definición de la arquitectura GCN (Piston) y del paso por época (Injector).
    └── Plotter.py         # Evaluador estadístico y generador de reportes y gráficos de dispersión.
```

### 1.1. `run.py` (Punto de Entrada)
* **Ajustes de bajo nivel:** Modifica el entorno de ejecución fijando variables como `PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:512'` para evitar la fragmentación de la memoria en la GPU. Configura políticas de multiplicación de matrices habilitando TensorFloat-32 (`allow_tf32 = True`) en CUDA para acelerar el cómputo.
* **Descriptores del Sistema Operativo:** Utiliza el módulo nativo `resource` para elevar el límite de archivos abiertos (`RLIMIT_NOFILE`) hasta un máximo seguro (ej. 65536). Esto previene fallos fatales de tipo *“Too many open files”* causados por el multiprocesamiento al cargar múltiples grafos en paralelo de forma concurrente.
* **Diagnóstico e Inicialización:** Verifica y reporta el hardware disponible (GPU, versión de CUDA, versión de PyTorch y VRAM total) antes de cargar el archivo de configuración JSON pasado como argumento por la línea de comandos (`sys.argv[1]`) e invocar al `Manager`.

### 1.2. `Manager.py` (Clase `V05` - Versión Actual: v1.3.3)
* **Validación y Casteo:** Lee los parámetros procedentes del archivo JSON y aplica una validación de tipos estricta para asegurar que las listas, booleanos y valores numéricos estén en rangos correctos antes de iniciar el entrenamiento.
* **Control del Flujo Operacional:** Dirige las ejecuciones decidiendo entre dos caminos principales:
    1.  **Búsqueda en Rejilla (Grid Search):** Si las optimizaciones bayesianas están apagadas, recorre secuencialmente bucles anidados basados en listas estáticas de control (ej. `learning_rates_list`, `number_of_layers_list`).
    2.  **Optimización Bayesiana (Optuna):** Si están encendidas, inicializa estudios basados en estimadores de Parzen estructurados en árbol (TPE). Declara las funciones objetivo (`objective_features`, `objective_architecture`, `objective_hyperparameters`) que sugieren parámetros y evalúan la pérdida resultante.
* **Persistencia y Telemetría:** Centraliza el inicio y cierre estructurado de las ejecuciones en **Weights & Biases (WandB)** y gestiona el guardado del historial de Optuna en una base de datos SQLite local para habilitar paneles interactivos con `optuna-dashboard`.

### 1.3. `DataHandler.py` (Versión Actual: v1.1.0)
* **Extracción Paralela:** El subcomponente `DataDealer` utiliza un `multiprocessing.Pool` para distribuir el análisis estructural de las cadenas SMILES a través de múltiples hilos de la CPU, agilizando la conversión de texto a objetos gráficos tridimensionales de PyTorch Geometric.
* **Optimización del Pipeline de Datos:**
    * **Caché de Grafos:** Introduce almacenamiento en caché para las estructuras gráficas moleculares ya calculadas, evitando recrear los objetos entre diferentes épocas de entrenamiento o pruebas de hiperparámetros.
    * **Migración Estructural:** Sustituye el antiguo uso de `InMemoryDataset` por clases basadas en `TorchDataset` estándar para un consumo de memoria RAM más eficiente y predecible.
    * **Configuraciones Críticas de Carga:** Configura los DataLoaders geométricos con `pin_memory=True` (para anclar los tensores en la memoria RAM física acelerando la transferencia a la GPU) y fija un `prefetch_factor=3` junto con múltiples *workers* de carga, logrando que la GPU nunca sufra de inanición de datos (*starvation*).

### 1.4. `Normalizer.py` (Clase `Norm` - Versión Actual: v1.0.0)
* **Limpieza Química y Deduplicación:** Identifica y elimina registros moleculares duplicados evaluando posibles inconsistencias numéricas en sus valores objetivo. Utiliza **RDKit** para analizar la estructura química elemental, aislando y descartando automáticamente fragmentos inorgánicos o mezclas iónicas complejas que distorsionen los patrones de convolución.
* **Tratamiento Outlier e Inducción de Normalidad:** Aplica un filtro iterativo basado en el Rango Intercuartílico (IQR) para purgar valores atípicos de la propiedad. Posteriormente, evalúa transformaciones matemáticas de potencia (Box-Cox y Yeo-Johnson), utilizando la prueba de hipótesis `scipy.stats.normaltest` de SciPy. Selecciona automáticamente la transformación matemática que proporcione el valor $p$ más alto (garantizando que la variable objetivo se asemeje lo máximo posible a una distribución normal/gaussiana para estabilizar los gradientes).
* **Mecanismo Inverso:** Preserva la función matemática de detransformación para permitir devolver más tarde las predicciones del modelo a su escala fisicoquímica real.

### 1.5. `Itinerary.py` (Clase `Train` - Versión Actual: v1.1.0)
* **Gestión del Ciclo de Vida:** Controla el bucle de épocas pasándole los lotes de datos al motor matemático. Monitorea las métricas en el conjunto de validación interna al terminar cada época y actualiza los planificadores dinámicos de tasa de aprendizaje (*Schedulers*).
* **Optimización de GPU:** Ejecuta las predicciones de evaluación y prueba envolviéndolas estrictamente en bloques `torch.inference_mode()` (reemplazando al antiguo `torch.no_grad()`), lo que suprime el rastreo de gradientes y ahorra cantidades significativas de VRAM. Los cargadores de evaluación se sitúan fuera de los bucles de entrenamiento principales para evitar fugas de memoria.
* **Desactivación de `torch.compile`:** El framework desactiva explícitamente el método de compilación nativo de PyTorch 2.x debido a que las moléculas poseen un número variable de átomos y enlaces, provocando tamaños de grafos y matrices altamente dinámicos que causan una recompilación constante en cada lote (*graph breaks*), ralentizando el entrenamiento.
* **Mecanismo de Rescate (EMERGENCY):** Envuelve la fase ordinaria de guardado en bloques `try-except`. Si ocurre un fallo imprevisto al escribir gráficos o tablas estadísticas en el almacenamiento local, el sistema captura el error, crea inmediatamente un directorio llamado `EMERGENCY` en la raíz, genera un archivo `.csv` con la configuración exacta utilizada y salva un volcado de los pesos internos del modelo en un archivo `.pth`, salvaguardando el tiempo de computación invertido.

### 1.6. `Engine.py` (Versión Actual: v1.0.2)
* **`Injector` (Entrenador por Lote):** Administra los pasos de cálculo puros (propagación hacia adelante, cómputo de la pérdida con el criterio seleccionado, retropropagación y actualización de pesos mediante el optimizador). Para maximizar la velocidad en la tarjeta gráfica, elimina por completo el uso de llamadas síncronas bloqueantes tipo `.item()` en los bucles internos de lotes, sustituyéndolas por extracciones asíncronas seguras mediante el método `.detach()`.
* **`Piston` (Arquitectura Convolucional de Grafos GCN):**
    * **Construcción Dinámica:** Instancia bloques compuestos por una capa de convolución espacial sobre grafos (`GCNConv`), seguida de una capa de normalización por lote unidimensional (`BatchNorm1d`), funciones de activación no lineales parametrizables y una capa de abandono regularizado (`Dropout(p=0.2)`). El número de capas y neuronas se ajusta dinámicamente según lo indicado por la rejilla o por Optuna.
    * **Agregación Molecular Global (Pooling):** Reúne las características individuales de los átomos para generar una única firma numérica para toda la molécula. Admite tres modos configurables: promedio global (`mean`), máximo global (`max`) o una agregación compuesta paralela que concatena ambos vectores (`both`) duplicando el tamaño de entrada de la capa lineal final de regresión (`Linear`).

### 1.7. `Plotter.py` (Clase `AllStats` - Versión Actual: v1.0.2)
* **Procesamiento Estadístico:** Recibe los tensores crudos de predicciones, los procesa a través del detransformador matemático de `Normalizer` para devolverlos a sus magnitudes físicas correctas, y calcula métricas científicas estandarizadas: Coeficiente de Determinación ($R^2$), Error Absoluto Medio (MAE) y Error Cuadrático Medio (MSE).
* **Visualización:** Genera gráficos de dispersión avanzados comparando los valores experimentales reales frente a las predicciones del modelo empleando las librerías *Seaborn* y *Matplotlib*. Adiciona una columna identificadora de pliegues (`fold`) para discriminar los resultados procedentes de validaciones cruzadas recurrentes.
* **Mecanismo de Restricción (`plotting_resrictor`):** Implementa un filtro lógico condicionado por el parámetro `"minimum_best_score"`. Si el rendimiento final del modelo evaluado no supera este umbral mínimo de precisión, el módulo bloquea preventivamente la escritura de imágenes y archivos tabulares en el disco duro, evitando saturar el almacenamiento local con resultados de entrenamientos basura o arquitecturas mediocres.

---

## 2. Guía de Configuración Detallada del Archivo `setup.json`

El archivo `setup.json` centraliza el control operativo total del framework. A continuación se detallan todas sus propiedades agrupadas funcionalmente.

### 2.1. Configuración del Dataset y Entrada de Datos
* **`"dataset_treatment"`** (`Boolean`): Si está en `true`, procesa el archivo CSV a través de las rutinas de limpieza, remoción de outliers e inducción de normalidad del módulo `Normalizer.py`. Si es `false`, los datos pasan crudos directamente al convertidor de grafos.
* **`"column_name_of_id"`** (`String`): Nombre exacto de la columna en el CSV que funge como el identificador único para cada fila. Ejemplo: `"ID"`.
* **`"column_name_of_property"`** (`String`): Nombre exacto de la columna objetivo continua que se desea predecir. Ejemplo: `"SOL"`.
* **`"column_name_of_smiles"`** (`String`): Nombre exacto de la columna que almacena la representación en cadena de texto molecular SMILES. Ejemplo: `"SMILES"`.
* **`"dataset_path"`** (`String`): Ruta física relativa o absoluta hacia el archivo del dataset en formato CSV. Ejemplo: `"./datasets/data.csv"`.

### 2.2. División del Dataset y Ciclo de Vida Base
* **`"test_dataset_ratio"`** (`Float`): Proporción de datos del dataset original extraídos y aislados estrictamente para conformar el conjunto de prueba externo (*test set*), el cual permanece oculto para el modelo durante todo el proceso de entrenamiento. Rango válido: `0.0` a `1.0`. Ejemplo: `0.1` ($10\%$).
* **`"validation_dataset_ratio"`** (`Float`): Proporción de los datos de entrenamiento remanentes que se destinarán al conjunto de validación interna (*validation set*) para evaluar la pérdida entre épocas, actualizar *schedulers* e indicar el Early Stopping. Ejemplo: `0.1` ($10\%$).
* **`"number_of_epochs"`** (`Integer`): Límite máximo de épocas completas de entrenamiento permitidas por cada iteración del modelo en caso de no activarse previamente la parada temprana. Ejemplo: `1500`.
* **`"plot_best_trials"`** (`Boolean`): Controla si el sistema debe forzar la generación y guardado de gráficos estadísticos intermedios para las pruebas individuales evaluadas en los ciclos de optimización bayesiana.

### 2.3. Mecanismo de Parada Temprana (Early Stopping)
* **`"early_stopping"`** (`Boolean`): Activa o desactiva la interrupción prematura del ciclo de entrenamiento si el modelo deja de mostrar mejoras significativas.
* **`"early_stop_handicap"`** (`Integer`): Ventana de épocas iniciales de "gracia". Durante este periodo el monitor de Early Stopping permanece congelado, permitiendo al optimizador superar la inestabilidad matemática inicial sin consumir la paciencia del contador. Ejemplo: `50`.
* **`"early_stop_patience"`** (`Integer`): Cantidad máxima de épocas consecutivas permitidas sin observar una disminución neta en la pérdida del conjunto de validación antes de abortar el entrenamiento del modelo. Ejemplo: `150`.

### 2.4. Control de Listas Estáticas (Búsqueda en Rejilla - Grid Search)
*Nota: Estas listas se recorren de forma secuencial mediante bucles anidados si los módulos de optimización bayesiana están desactivados (`false`).*
* **`"random_states_list"`** (`List[int]`): Lista de semillas enteras utilizadas para fijar la partición aleatoria y la reproducibilidad en la división de los conjuntos de datos. Ejemplo: `[0]`.
* **`"random_seeds_list"`** (`List[int]`): Lista de semillas enteras para inicializar de manera reproducible las matrices de pesos iniciales de las capas de la red neuronal. Ejemplo: `[0]`.
* **`"learning_rates_list"`** (`List[float]`): Lista de tasas de aprendizaje estáticas iniciales para evaluar en el optimizador. Ejemplo: `[0.01, 0.001, 0.0001]`.
* **`"batch_sizes_list"`** (`List[int]`): Lista de tamaños de lote de datos para el entrenamiento estático. Ejemplo: `[256]`.
* **`"number_of_layers_list"`** (`List[int]`): Profundidades fijas de la red (número de capas convolucionales GCN consecutivas) a explorar en la rejilla. Ejemplo: `[3, 4]`.
* **`"number_of_neurons_list"`** (`List[int]`): Número de canales ocultos o neuronas por capa convolucional aplicados de manera fija en el barrido de la rejilla. Ejemplo: `[256]`.

### 2.5. Programadores de Tasa de Aprendizaje (Schedulers)
* **`"learning_rate_scheduler"`** (`Boolean`): Determina si se acopla un gestor dinámico sobre la tasa de aprendizaje del optimizador a lo largo del entrenamiento.
* **`"cosine_annealing_scheduler"`** (`Boolean`): Si se establece en `true`, el framework prioriza un programador de recocido cosenoidal con reinicios cálidos (`CosineAnnealingWarmRestarts`). Si se define en `false`, el sistema recurre por defecto al descenso por estancamiento en mesetas (`ReduceLROnPlateau`), reduciendo el valor de la tasa ante periodos prolongados de nula mejora.
* **`"scheduler_factor"`** (`Float`): Factor multiplicativo de penalización utilizado por `ReduceLROnPlateau`. Ejemplo: `0.5` (disminuye la tasa de aprendizaje exactamente a la mitad).
* **`"scheduler_patience"`** (`Integer`): Número de épocas de espera sin mejoras en validación que tolera el scheduler antes de aplicar la corrección del factor numérico. Ejemplo: `30`.

### 2.6. Validación Cruzada, Restricciones y Monitoreo Remoto
* **`"minimum_best_score"`** (`Float/Null`): Puntuación o pérdida mínima requerida como corte de calidad. Si el rendimiento del modelo evaluado es peor que este valor, el framework restringe la creación de archivos gráficos y tablas estadísticas a través de `Plotter.py` para proteger el disco local de saturación. Ejemplo: `0.2`.
* **`"cross_validation"`** (`Boolean`): Habilita el re-entrenamiento completo bajo una partición por pliegues del modelo seleccionado para certificar de forma robusta su capacidad de generalización estadística.
* **`"k_folds"`** (`Integer`): Número total de pliegues en los que se segmentará el dataset de entrenamiento para la validación cruzada. Ejemplo: `10`.
* **`"use_wandb"`** (`Boolean`): Habilita el inicio, la carga y la sincronización en tiempo real de los logs de entrenamiento e hiperparámetros hacia la plataforma en la nube **Weights & Biases**.
* **`"project_name"`** (`String`): Nombre alfanumérico global asignado al espacio de trabajo o proyecto dentro de WandB y del registro persistente de Optuna. Ejemplo: `"01234567sol256"`.
* **`"use_optuna_db"`** (`Boolean`): Si es `true`, escribe el historial y las trazas completas de las pruebas de optimización bayesiana en un archivo de base de datos SQLite local (`optuna.db`), permitiendo auditar visualmente las superficies de optimización mediante el comando `optuna-dashboard optuna.db`.
* **`"model"`** (`String`): Especifica la clase o tipo primitivo de capa convolucional geométrica de grafos que el motor inyectará internamente en la red. Ejemplo: `"GCNConv"`.

---

## 3. Diccionarios de Mapeo Numérico Interno

Para mantener la ligereza en la estructura del archivo JSON, los componentes complejos de software y los descriptores químicos moleculares se ingresan en `setup.json` mediante códigos numéricos enteros mapeados internamente por `Manager.py`:

### 3.1. Descriptores Atómicos de Nodos (`"node_features"`)
Recibe una lista de números enteros que determina qué propiedades específicas de cada átomo serán extraídas utilizando la librería quimioinformática RDKit para estructurar el vector inicial de características del nodo en el grafo molecular:
* **`0`**: Número Atómico Elemental (`AtomNum`).
* **`1`**: Grado de Conectividad o número total de enlaces del átomo (`Degree`).
* **`2`**: Carga Formal neta asignada al átomo (`FormalCh`).
* **`3`**: Tipo de Hibridación de orbitales electrónicos del átomo (`Hybridization`).
* **`4`**: Indicador binario de si el átomo forma parte de una estructura o anillo aromático (`Aromatic`).
* **`5`**: Conteo total de átomos de hidrógeno implícitos o explícitos unidos al átomo (`NumHs`).
* **`6`**: Indicador binario de si el átomo está contenido dentro de algún tipo de anillo químico (`InRing`).
* **`7`**: Estimación de carga parcial calculada mediante el método cuantitativo de Gasteiger (`GastCh`).

*Ejemplo en JSON para activar todas las características:* `"node_features": [0, 1, 2, 3, 4, 5, 6, 7]`

### 3.2. Algoritmos de Optimización de Pesos (`"optimizers_list"`)
Lista de identificadores para seleccionar los optimizadores nativos de PyTorch encargados de actualizar las matrices de pesos de la red:
* **`0`**: `Adadelta`
* **`1`**: `Adagrad`
* **`2`**: `Adam`
* **`3`**: `AdamW` (Adam con desacoplamiento de decaimiento de pesos).
* **`4`**: `Adamax`
* **`5`**: `NAdam` (Adam con gradiente acelerado de Nesterov).
* **`6`**: `RAdam` (Adam Rectificado).
* **`7`**: `RMSprop`
* **`8`**: `Rprop`

*Ejemplo en JSON:* `"optimizers_list": [2, 3]` *(Explora Adam y AdamW).*

### 3.3. Funciones de Pérdida o Criterios de Error (`"loss_functions_list"`)
Lista de identificadores para fijar el criterio matemático de error que el optimizador buscará minimizar en el entrenamiento:
* **`0`**: `L1Loss` (Error Absoluto Medio - MAE).
* **`1`**: `MSELoss` (Error Cuadrático Medio).
* **`2`**: `HuberLoss` (Pérdida Huber suave, robusta y menos sensible ante la presencia de valores atípicos remanentes).
* **`3`**: `SmoothL1Loss` (Variación matemática suave del error absoluto L1).

*Ejemplo en JSON:* `"loss_functions_list": [0, 1, 2, 3]`

### 3.4. Funciones de Activación No Lineal (`"activation_functions_list"`)
Lista de identificadores de las funciones de activación que se inyectarán de forma intermedia entre las capas convolucionales consecutivas del modelo:
* **`0`**: `ELU`
* **`1`**: `GELU`
* **`2`**: `SiLU`
* **`3`**: `Mish`
* **`4`**: `LeakyReLU`
* **`5`**: `ReLU` (Unidad de Rectificación Lineal ordinaria).
* **`6`**: `Sigmoid` (Uso restringido o específico).
* **`7`**: `Tanh` (Tangente Hiperbólica).

*Ejemplo en JSON:* `"activation_functions_list": [5]` *(Fija el uso de ReLU).*

---

## 4. Motores de Optimización Bayesiana (Optuna)

El framework incorpora tres motores inteligentes e independientes de búsqueda basados en Optuna y algoritmos de Estimación de Parzen Estructurados en Árbol (TPE). Al activarse alguno de ellos (`true`), el sistema ignora las listas fijas de la búsqueda en rejilla para realizar muestreos inteligentes.

### 4.1. Optimización de Hiperparámetros Clínicos
Automatiza la búsqueda de la tasa de aprendizaje, el tamaño de lote, el optimizador y la función de pérdida ideal para la convergencia.
* **`"hyperparameter_optimizer"`** (`Boolean`): Habilita o deshabilita este motor específico de búsqueda bayesiana de hiperparámetros básicos.
* **`"hyperparameter_optimizer_trials"`** (`Integer`): Número máximo de combinaciones o ejecuciones independientes individuales que realizará Optuna para este estudio. Ejemplo: `100`.
* **`"hyperparameter_optimizer_timeout"`** (`Integer/Null`): Límite de tiempo absoluto expresado en segundos tras el cual el estudio finalizará de forma limpia salvando los mejores resultados obtenidos hasta el momento. Si se declara `null`, no hay límite de tiempo.
* **`"hyperparameter_optimizer_patience"`** (`Integer/Null`): Cantidad máxima de intentos consecutivos en los que Optuna puede fallar o no encontrar mejoras en la pérdida antes de congelar y detener el estudio de forma anticipada. Ejemplo: `30`.
* **`"batch_size_range"`** (`List[int]`): Intervalo cerrado que delimita los valores mínimos y máximos enteros para el tamaño de lote que el optimizador puede muestrear en escala logarítmica. Ejemplo: `[8, 128]`.
* **`"learning_rate_range"`** (`List[float]`): Intervalo cerrado que delimita los límites mínimos y máximos continuos para la tasa de aprendizaje inicial evaluada en escala logarítmica. Ejemplo: `[10e-4, 10e-1]`.
* **`"force_batch_size_value"`** (`Integer/Null`): Si se le asigna un número entero, el framework anula por completo el muestreo dinámico del tamaño de lote del rango anterior, forzando y fijando dicho tamaño durante toda la campaña de optimización bayesiana de hiperparámetros. Ejemplo: `25`.

### 4.2. Optimización del Perfil Arquitectónico (Ancho y Profundidad)
Permite delegar en Optuna el diseño estructural de la red convolucional profunda de grafos.
* **`"architecture_optimizer"`** (`Boolean`): Si se establece en `true`, faculta a Optuna para sugerir de forma dinámica la profundidad del modelo (es decir, la cantidad exacta de capas convolucionales GCN consecutivas a apilar).
* **`"optimize_width"`** (`Boolean`): Si se define en `true`, habilita que el número de neuronas o canales ocultos por capa varíe dinámicamente según un muestreo polinomial independiente en cada nivel, permitiendo perfiles geométricos variables (redes piramidales, de cuello de botella, etc.).
* **`"width_optimization_space"`** (`List[int]`): Límites mínimo y máximo de neuronas permitidas para explorar en las capas de la red. Ejemplo: `[16, 256]`.
* **`"architecture_optimizer_trials"`** / **`"timeout"`** / **`"patience"`**: Parámetros de control de ciclo idénticos a los descritos en la sección anterior, pero aplicados exclusivamente al estudio de búsqueda estructural de la red.

### 4.3. Optimización de Descriptores Químicos de Entrada (Feature Optimization)
Este motor realiza una búsqueda combinatoria binaria de tipo "encendido/apagado" sobre la lista de descriptores de átomos disponibles. Su objetivo es identificar qué subconjunto óptimo de características atómicas maximiza el poder de generalización de la GCN, descartando descriptores redundantes, altamente correlacionados o que inyecten ruido matemático al grafo de entrada.
* **`"feature_optimization"`** (`Boolean`): Lanza el estudio bayesiano enfocado en encontrar la mejor combinación binaria de descriptores de nodos.
* **`"feature_optimizer_trials"`** / **`"timeout"`** / **`"patience"`**: Parámetros homólogos de control que regulan el límite de intentos, tiempo de expiración y tolerancia a la falta de mejoras durante el proceso de selección de descriptores químicos.
