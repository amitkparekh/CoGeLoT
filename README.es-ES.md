

# [Investigando el papel de la variedad de instrucciones y la dificultad de la tarea en tareas de manipulación robótica](https://arxiv.org/abs/2407.03967)

<a href="https://www.python.org/"><img alt="Python 3.11" src="https://img.shields.io/badge/Python 3.11-blue?logo=python&logoColor=white"></a>
<a href="https://pdm-project.org/en/latest/"><img alt="PDM" src="https://img.shields.io/badge/PDM-AC75D7?logo=pdm&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white"></a>
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-42b983)](https://detachhead.github.io/basedpyright)
[![CI](https://github.com/amitkparekh/CoGeLoT/actions/workflows/ci.yml/badge.svg)](https://github.com/amitkparekh/CoGeLoT/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2407.03967-b31b1b.svg)](https://arxiv.org/abs/2407.03967)


_Amit Parekh, Nikolas Vitsakis, Alessandro Suglia e Ioannis Konstas._

<br />

![Tabla de perturbaciones del artículo](docs/PERT%20Table.png)

_Revelando la verdadera robustez de los modelos multimodales: Un marco integral para explorar si los modelos son plausiblemente resilientes._




## Inicio rápido

> [!NOTE]
> Esta base de código descarga automáticamente puntos de control y conjuntos de datos, por lo que no necesitas hacerlo manualmente. Todo está alojado en Hugging Face y usa HF, por lo que también queda en caché.

1. Clona este repositorio y navega al directorio

    ```bash
    git clone https://github.com/amitkparekh/CoGeLoT.git
    cd CoGeLoT
    ```

2. Instala las dependencias (usé [PDM](https://pdm-project.org/en/latest/) y Python 3.11)

    ```bash
    pdm install
    ```

3. Asegúrate de que todo funcione y esté instalado correctamente

    ```bash
    pdm run pytest --deselect tests/test_online_evaluation.py
    ```

4. Entrena un modelo

    ```bash
    pdm run python src/cogelot/entrypoints/train.py experiment=01_their_vima
    ```

5. Evalúa un modelo a partir de uno de los [puntos de control proporcionados](#model-architectures-and-checkpoints)

    ```bash
    pdm run python src/cogelot/entrypoints/evaluate.py trainer.devices=1 model.model.wandb_run_id=8lkml12g
    ```

  Para `model.model.wandb_run_id`, puedes usar cualquiera de los Run IDs de la [tabla inferior](#model-architectures-and-checkpoints).



## Contenido


> [!NOTE]
> Este proyecto tiene el nombre en código `cogelot`, por lo que así se llama la biblioteca para evitar tener que reescribir todo.

- [¿Qué está incluido en este proyecto?](#what-is-included)
    - [Arquitecturas de modelos y puntos de control proporcionados](#model-architectures-and-checkpoints)
- [Cómo ejecuté las cosas](#how-i-ran-things)
    - [Instalación de dependencias](#how-i-managed-and-installed-dependencies)
    - [Verificación fácil de que todo funciona](#how-i-checked-that-everything-worked-before-i-ran-things)
    - [Entrenamiento de modelos](#how-i-trained-models)
    - [Ejecución de puntos de control en el entorno](#how-i-ran-checkpoints-in-the-environment)
    - [Preparación del conjunto de datos](#how-i-prepared-the-dataset)
- [Licencia](#license)
- [Cita](#citation)





## ¿Qué está incluido?

Todo. Deberías poder ejecutar cada experimento del artículo. Los conjuntos de datos y los modelos están alojados en HF.

Aunque intenté dejar todo claro y al frente, algunas cosas podrían estar ocultas. Si crees que estas cosas deberían resaltarse más, ¡siéntete libre de abrir un PR y traerlas al frente! Definitivamente tendré en cuenta opiniones sobre esto para futuros proyectos.

Además, he intentado trabajar de manera restringida, limpia y robusta. Espero que te ayude tanto como me ayudó a mí.



### Arquitecturas de modelos y puntos de control

A continuación se muestra una tabla de cada ejecución de modelo y dónde encontrar los puntos de control. Estamos proporcionando todos los puntos de control al final de cada época, aunque solo usamos el de la última época.

**No necesitas descargar los puntos de control manualmente.** Esta biblioteca contiene múltiples métodos y funciones para que los puntos de control funcionen en nuestro marco de trabajo, y todo está incluido para ti. Todos los puntos de control de los modelos están almacenados en Hugging Face, pero *no funcionarán con la biblioteca Transformers fuera de la caja*.



| Estilo de instrucción | Modalidades de instrucción | Condicionamiento del prompt | Codificador visual | ¿Objetos barajados? | ID de ejecución de WandB | ID de experimento |
|:--|:--|:--|:--|:--:|:--:|:---|
| Original | Texto + Visual | Atención cruzada | Centrado en objetos | Falso | [`8lkml12g`](https://huggingface.co/amitkparekh/cogelot/tree/main/8lkml12g) | `01_their_vima` |
| Original | Texto + Visual | Atención cruzada | Centrado en objetos | Verdadero | [`ftwoyjb1`](https://huggingface.co/amitkparekh/cogelot/tree/main/ftwoyjb1) | `01_their_vima_shuffle_obj` |
| Original | Texto + Visual | Atención cruzada | Parches de imagen | N/A | [`ln4nrqhg`](https://huggingface.co/amitkparekh/cogelot/tree/main/ln4nrqhg) | `01_their_vima_patches` |
| Original | Texto + Visual | Concatenar | Centrado en objetos | Falso | [`bhuja4vo`](https://huggingface.co/amitkparekh/cogelot/tree/main/bhuja4vo) | `08_their_gpt` |
| Original | Texto + Visual | Concatenar | Centrado en objetos | Verdadero | [`wn9jc5l8`](https://huggingface.co/amitkparekh/cogelot/tree/main/wn9jc5l8) | `08_their_gpt_shuffle_obj` |
| Original | Texto + Visual | Concatenar | Parches de imagen | N/A | [`efxugme9`](https://huggingface.co/amitkparekh/cogelot/tree/main/efxugme9) | `08_their_gpt_patches` |
| Parafraseos | Texto + Visual | Atención cruzada | Centrado en objetos | Falso | [`2df3mwfn`](https://huggingface.co/amitkparekh/cogelot/tree/main/2df3mwfn) | `02_their_vima` |
| Parafraseos | Texto + Visual | Atención cruzada | Centrado en objetos | Verdadero | [`0nsnkaer`](https://huggingface.co/amitkparekh/cogelot/tree/main/0nsnkaer) | `02_their_vima_shuffle_obj` |
| Parafraseos | Texto + Visual | Atención cruzada | Parches de imagen | N/A | [`ah5btw8w`](https://huggingface.co/amitkparekh/cogelot/tree/main/ah5btw8w) | `02_their_vima_patches` |
| Parafraseos | Texto + Visual | Concatenar | Centrado en objetos | Falso | [`fs5v61mz`](https://huggingface.co/amitkparekh/cogelot/tree/main/fs5v61mz) | `09_their_gpt` |
| Parafraseos | Texto + Visual | Concatenar | Centrado en objetos | Verdadero | [`xb3yttg9`](https://huggingface.co/amitkparekh/cogelot/tree/main/xb3yttg9) | `09_their_gpt_shuffle_obj` |
| Parafraseos | Texto + Visual | Concatenar | Parches de imagen | N/A | [`zby6xk27`](https://huggingface.co/amitkparekh/cogelot/tree/main/zby6xk27) | `09_their_gpt_patches` |



## Cómo ejecuté las cosas


> [!IMPORTANT]
> **Todo lo que se ejecutó, en alguna forma, comienza desde un módulo en `src/cogelot/entrypoints/`.** Esto es lo que se usó para ejecutar la creación del conjunto de datos, entrenar modelos, evaluar modelos y más. Todo lo que ejecuté comenzó desde esa carpeta, cada vez.


Esta no es una biblioteca integral hecha para todos los casos de uso y cada posible escenario. Es un proyecto de investigación. Dicho esto, intenté hacer todo lo más claro posible para ti. En esta sección, detallé cómo hice todo para que puedas usarlo como ejemplo de cómo empezar tú mismo.

He intentado asegurarme de que las docstrings y comentarios sean relevantes y detallados. Si quieres más información sobre lo que hace una función o por qué lo hace, siéntete libre de abrir un issue. Si descubres algo que no he descrito lo suficiente, siéntete libre de hacer un PR mejorando mi documentación para que tú, yo y las personas futuras podamos beneficiarnos de tu contribución.



### Cómo gestioné e instalé las dependencias


Usé [PDM](https://pdm-project.org/en/latest/) para gestionar este proyecto. Todo lo que necesitas saber sobre la instalación de las dependencias para este proyecto se encuentra en `pyproject.toml`.

Para instalar rápidamente y comenzar, puedes ejecutar lo siguiente:

```bash
pdm install
```


<details>
<summary><b>¿Qué pasa si usas `requirements.txt`?</b></summary>

He exportado e incluido el `requirements.txt` desde PDM. El uso de este archivo depende de ti. No lo mantendré actualizado, pero está disponible si lo necesitas.

</details>

<details>
<summary><b>Cómo instalo dependencias en cada máquina</b></summary>

Literalmente solo ejecuto lo siguiente en las máquinas que uso. No uso Windows, así que no puedo ayudarte en ese sentido.

```bash
mise use python@3.11 pdm@latest
pdm install
```
</details>


<details>
<summary><b>Cómo asegurarte de que funcione en tu máquina</b></summary>

La forma más rápida de asegurarte de que todo está configurado es ejecutar cualquiera de los siguientes:

- Si sabes que tienes un entorno virtual activado o similar
    ```bash
    python -m cogelot
    ```

- Si estás usando PDM en lugar de activar el entorno virtual
    ```bash
    pdm run python -m cogelot
    ```

</details>



### Cómo verifiqué que todo funcionaba antes de ejecutar las cosas

Las cosas pasan y las cosas fallan. Necesitaba una verificación rápida para asegurarme de que todo funcionaba. Desarrollé todo usando pruebas para verificar que cada pieza funciona de forma aislada y conjunta. Esto es lo primero que hice al usar una máquina, nodo o lo que sea nueva.

Puedes encontrar todas las pruebas en la carpeta `tests/`. Las diversas pruebas son una buena manera de ver cómo se implementaron y usaron las diferentes piezas. Aunque la cobertura no es del 100%, usé las pruebas con puntos de interrupción para verificar que todo funciona como se espera.

<details>
<summary><b>Cómo asegurarte de que todas las pruebas se carguen sin errores</b></summary>

```bash
pdm run pytest --deselect tests/test_online_evaluation.py --collect-only
```

Esto también es útil para simplemente asegurarte de que las cosas se instalaron correctamente y de que todas las pruebas se pueden encontrar.

</details>

<details>
<summary><b>Cómo ejecutar todas las pruebas</b></summary>

```bash
pdm run pytest --deselect tests/test_online_evaluation.py
```

También tengo CI haciendo esto, por lo que puedes verificar el badge en la parte superior del README para ver si todo funciona como se espera.

Echa un vistazo a [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) si quieres saber más sobre ejecutar pruebas en paralelo, o simplemente añade ` -n auto` al final de los comandos anteriores. Lo hace más rápido.[^1]

[^1]: No sé qué pasa si reemplazas `auto` con un número que tenga más procesos que tu máquina. Tal vez no lo hagas.

</details>


> [!TIP]
> Antes de iniciar una instancia y comenzar a entrenar con GPUs, puedes ejecutar el comando anterior en tu máquina para asegurarte de que todo funciona en CPU. Como Lightning maneja toda la comunicación de GPU, si funciona en CPU, hay un 99% de probabilidad de que funcione en GPU.[^2]


[^2]: Este número es inventado, pero estoy bastante seguro de ello.



### Cómo entrené modelos

> [!IMPORTANT]
> **Todo el entrenamiento de modelos fue orquestado con [Hydra](https://hydra.cc/), y se puede encontrar en la carpeta `configs/`.**

Me esforcé mucho con Hydra y todo es bastante compositivo. El subdirectorio `configs/experiments` contiene los experimentos que se ejecutaron (y se conectan directamente con la tabla de puntos de control). Como resultado, si solo quieres entrenar un modelo, puedes ejecutar:


```bash
pdm run python src/cogelot/entrypoints/train.py experiment=01_their_vima
```


> [!TIP]
> Puedes encontrar los experimentos en la carpeta, o verificar la columna `Experiment ID` en la [tabla anterior](#model-architectures-and-checkpoints) para saber qué significa cada uno, ya que los nombres no son los más claros.



<details>
<summary><b>Entrenamiento en diferentes hardware</b></summary>

La carpeta `configs/hardware` contiene las configuraciones de hardware que se usaron para ejecutar los experimentos. Estas se usan para establecer el número de GPUs, el número de CPUs y la memoria disponible para el modelo. Estos estaban preestablecidos para el clúster que usaba, pero puedes ajustarlos a tus necesidades.

</details>



<details>
<summary><b>Cómo entrenar modelos en OCI</b></summary>

Esto fue hace un tiempo, pero tenía un script de configuración que puedes encontrar en `scripts/setup-oci-a100.sh`. Esto se usó para configurar el entorno en la instancia de OCI que usaba. No es perfecto, pero es un buen punto de partida.

</details>



<details>
<summary><b>Cómo entrené modelos en K8s</b></summary>

Ejecutar en K8s fue un poco más complejo, pero todo está aquí. Dicho esto, será diferente para tus configuraciones.

Mi especificación de pod fue:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: &name cogelot-1
  namespace: ???
spec:
  restartPolicy: Never
  containers:
    - name: 1st
      image: amitkparekh/python-pdm-cuda:latest
      envFrom:
        - secretRef:
            name: amit-cogelot
      imagePullPolicy: Always
      command: ["/bin/bash", "-c"]
      args:
        - gh repo clone amitkparekh/cogelot cogelot &&
          cd cogelot &&
          bash ./scripts/setup-eidf.sh 2>&1 | tee setup-eidf.log &&
          sleep infinity
          #bash ./scripts/run-sweep-4.sh
      resources:
        requests:
          cpu: &num-cpu 10
          memory: &num-memory "150Gi"
          nvidia.com/gpu: &num-gpu 4
        limits:
          cpu: *num-cpu
          memory: *num-memory
          nvidia.com/gpu: *num-gpu
      volumeMounts:
        - mountPath: /mnt/ceph_rbd
          name: volume
          # this is necessary for training in distributed mode - used for different processes to communicate
        - mountPath: /dev/shm
          name: dshm1
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
  volumes:
    - name: volume
      persistentVolumeClaim:
        claimName: *name
    - name: dshm1
      emptyDir:
        medium: Memory
```


El Dockerfile es público y los secretos contenían lo siguiente:

```
WANDB_API_KEY=???
HUGGING_FACE_HUB_TOKEN=???
GH_TOKEN=???

HF_HUB_VERBOSITY=info

WANDB_CONFIG_DIR=/mnt/ceph_rbd/wandb
WANDB_CACHE_DIR=/mnt/ceph_rbd/wandb
HF_HOME=/mnt/ceph_rbd/huggingface
TORCH_HOME=/mnt/ceph_rbd/torch
```

</details>



### Cómo ejecuté puntos de control en el entorno

Nuevamente, esto usa Hydra, por lo que, como el entrenamiento, el punto de entrada es `src/cogelot/entrypoints/evaluate.py` y la configuración para ello es `configs/evaluate.yaml`.

Para ejecutar la evaluación en el entorno, usé el siguiente comando:

```bash
pdm run python src/cogelot/entrypoints/evaluate.py trainer.devices=1 model.model.wandb_run_id=8lkml12g
```


<details>
<summary><b>Cómo elegir tu punto de control</b></summary>

El parámetro `model.model.wandb_run_id` es importante y se usa para obtener el punto de control a evaluar. El ID del punto de control es el de la tabla anterior.

Por defecto, usamos la época del último punto de control, pero si quieres cambiar la época, simplemente añade `model.model.epoch=<número_de_época>` al comando.

</details>

<details>
<summary><b>Cómo ejecutar múltiples ejecuciones en paralelo</b></summary>

`trainer.devices` crea múltiples procesos de CPU para la evaluación, ya que la evaluación no necesita GPU.
Cambia el número en el comando según cuántos procesos quieras.

Cosas importantes a tener en cuenta:

1. Cuantos más procesos/dispositivos uses, más memoria necesitarás, ya que múltiples instancias del modelo se cargan en la memoria.
2. No hice nada especial con el batching entre instancias. Dado que usamos CPU para la evaluación, no fue necesario.

</details>



<details>
<summary><b>Cómo perturbar las instrucciones</b></summary>

Puedes encontrar todos estos en `configs/evaluation_instance_transform/`. Para cada nombre de archivo, puedes invocarlos añadiendo `evaluation_instance_transform=<nombre_archivo>` al comando.

| Transformación de instancia de evaluación | Descripción |
|:--|:--|
| `noop` | Intercalar modalidades en el prompt, *por defecto* |
| `gobbledygook_tokens` | Aplicar *Tokens Gobbledygook* al prompt |
| `gobbledygook_words` | Aplicar *Palabras Gobbledygook* al prompt |
| `reworded` | Usar instrucciones parafraseadas con modalidades intercaladas |
| `textual` | Convertir referentes visuales a texto |
| `textual_gobbledygook_words` | Convertir referentes visuales a texto y aplicar *Palabras Gobbledygook* |
| `textual_gobbledygook_tokens` | Convertir referentes visuales a texto y aplicar *Tokens Gobbledygook* |
| `textual_no_noun` | Convertir referentes visuales a texto, pero eliminar los sustantivos |
| `textual_no_texture` | Convertir referentes visuales a texto, pero eliminar las descripciones de los sustantivos |
| `textual_generic_noun` | Convertir referentes visuales a texto, pero reemplazar cada sustantivo con una forma genérica (ej. "block" se convierte en "thing") |

</details>


<details>
<summary><b>Cómo desactivar modalidades en el prompt</b></summary>


Puedes encontrar todos estos en `configs/evaluation_prompt_modality/`. Para cada nombre de archivo, puedes invocarlos añadiendo `evaluation_prompt_modality=<nombre_archivo>` al comando.

| Modalidad de prompt de evaluación | Descripción |
|:--|:--|
| `disable_none` | No hacer nada |
| `disable_text` | Desactivar la modalidad de texto |
| `disable_visual` | Desactivar la modalidad visual |
| `disable_both` | Desactivar ambas modalidades, básicamente enmascarando *cada token* |

</details>



<details>
<summary><b>Cómo permutar el orden de los tokens de objeto para las observaciones</b></summary>


Añade `model.should_shuffle_obj_per_observations=true` al comando. Esto barajará los tokens de objeto en la observación.

</details>



<details>
<summary><b>Cómo ejecutar con diferentes dificultades</b></summary>

Añade `model.difficulty=<dificultad>` al comando. Las dificultades son:

- `easy`
- `medium` _(sin usar)_
- `hard` _(sin usar)_
- `extreme`
- `distracting`
- `extremely_distracting`

</details>


<details>
<summary><b>Cómo ejecuté el punto de control de VIMA en el entorno</b></summary>

Descargué el punto de control desde el [repo de VIMA](https://github.com/vimalabs/VIMA), lo renomé a `them.ckpt` y lo coloqué en `storage/data/models`. Si quieres cambiar la ruta usada, puedes cambiarla en `configs/model/from_their_policy.yaml`.

Usé el siguiente comando para ejecutar el punto de control de VIMA en el entorno:

```bash
SLURM_JOB_NAME='bash' pdm run python src/cogelot/entrypoints/evaluate_theirs.py trainer.devices=20
```

Puedes usar todas las demás perturbaciones mencionadas arriba.

</details>



<details>
<summary><b>Cómo ejecutar puntos de control con una visualización en vivo</b></summary>

Si quieres ver qué está pasando en vivo, puedes añadir `environment@model.environment=display` al comando de evaluación.

Importante: **usa solo un proceso** porque no sé qué pasará si no lo haces.

Además, esto no se ejecutó en SLURM, solo en mi Mac. No puedo hablar por cada máquina, así que tu experiencia puede variar.

</details>

<details>
<summary><b>Cómo evaluar modelos en SLURM</b></summary>

Es muy poco probable que haya ejecutado las cosas en una sesión de tmux y simplemente las haya observado. No me gusta copiar y pegar cientos de comandos.

Como los experimentos a menudo se ejecutaron en un clúster de cómputo, ejecuté comandos con SLURM. Puedes encontrar estos archivos batch contenidos en `./scripts/slurm/`. Estos fueron hechos para mi sistema, por lo que es probable que se necesiten algunos ajustes, pero ¡espero que sea obvio y no demasiado complicado!

</details>


### Cómo preparé el conjunto de datos

Para que las cosas se ejecuten rápidamente, el conjunto de datos se cargó y analizó con Pydantic, y luego se convirtió en un [conjunto de datos HF](https://huggingface.co/datasets/amitkparekh/cogelot). Hay pruebas unitarias que muestran cómo se hizo esto en `tests/test_dataset_creation.py`.

El conjunto de datos se procesó en dos pasos. El primer paso fue analizar los datos sin procesar y serializarlos (pickle) en archivos individuales. Esto se hizo porque fue la parte más consume tiempo del proceso. El segundo paso fue cargar los archivos serializados y convertirlos en un conjunto de datos HF.

Para hacer la carga de datos eficiente durante el modelado, todas las instancias se tokenizaron de antemano. De manera similar, esto también está disponible en HF, con un nombre de configuración diferente.


> [!NOTE]
> Para cada uno de los siguientes comandos, puedes añadir `--help` para obtener más información sobre el comando, lo que hace y los diversos argumentos para controlarlo. Alternativamente, puedes cambiar las cosas usando los [ajustes de Pydantic](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) en `src/cogelot/commmon/settings.py`.
>
> Por ejemplo, cada comando tiene una forma de distribuir la carga a múltiples trabajadores, e incluso dividirlos en múltiples trabajos de SLURM para que vaya mucho más rápido.


<details>
<summary><b>Paso 1. Descargar los datos sin procesar de VIMA</b></summary>

Los datos sin procesar se descargaron desde VIMA. Cada instancia es una carpeta con múltiples archivos. Una vez extraídos, la estructura de carpetas se veía así:

```
<project_root>
└─ storage/
    └─ data/
        └─ raw/
            └─ vima_v6/
                └─ <task_name>/
                    └─ <instance_id>/
```

Usé enlaces simbólicos para facilitar la gestión de los datos, pero esta era la estructura de carpetas. Si quieres usar un directorio diferente, puedes cambiarlo usando los [ajustes de Pydantic](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) en `src/cogelot/common/settings.py`.

</details>


<details>
<summary><b>Paso 2. Analizar los datos originales</b></summary>

Los datos sin procesar se analizaron y serializaron en archivos individuales, y luego se convirtieron en un conjunto de datos HF. Esto es por velocidad.

```bash
pdm run python -m cogelot parse-original-dataset --replace-if-exists
pdm run python -m cogelot create-raw-dataset-per-task
```

Tengo archivos SBATCH separados para estos pasos:
  - `scripts/slurm/parse-original-dataset.sh`
  - `scripts/slurm/create-raw-dataset.sh`

</details>



<details>
<summary><b>Paso 3. Tokenizar y preprocesar para un entrenamiento más rápido</b></summary>

Nuevamente, preprocesamos y simplemente guardamos cada uno como pickles porque es más rápido antes de convertirlo en el conjunto de datos HF

```bash
pdm run python -m cogelot preprocess-instances
pdm run python -m cogelot create-preprocessed-dataset-per-task

```

Tengo archivos SBATCH separados para estos pasos:
  - `scripts/slurm/preprocess-instances.sh`
  - `scripts/slurm/create-preprocessed-dataset.sh`

</details>



<details>
<summary><b>Paso 4. Crear una variante del conjunto de datos con instrucciones parafraseadas</b></summary>

Usamos solo las instancias anteriores para hacer las nuevas variaciones, y usamos variables de entorno para crear las versiones preprocesadas del conjunto de datos.

```bash
pdm run python -m cogelot create-reworded-dataset-per-task original reworded

DATASET_VARIANT=reworded pdm run python -m cogelot preprocess-instances
DATASET_VARIANT=reworded pdm run python -m cogelot create-preprocessed-dataset-per-task
```

Nuevamente, tengo un archivo SBATCH para esto: `scripts/slurm/create-reworded-dataset.sh`, o más convenientemente, un script de bash para enviar trabajos SBATCH: `scripts/submit-reworded-dataset-creation-jobs.sh`.

</details>


<details>
<summary><b>Paso 5. Subir todos los conjuntos de datos</b></summary>

Este es solo para mí, pero consulta `scripts/submit-dataset-upload-jobs.sh` para subir todos los conjuntos de datos a HF lo más rápido posible sin alcanzar el límite de速率.

</details>



## Licencia

VIMA, VIMA-Bench y todos los artefactos de VIMA están licenciados bajo la Licencia MIT. Todo dentro de este repositorio continúa bajo la Licencia MIT.

## Cita

```bibtex
@misc{parekh2024investigatingroleinstructionvariety,
  title = {Investigating the {{Role}} of {{Instruction Variety}} and {{Task Difficulty}} in {{Robotic Manipulation Tasks}}},
  author={Amit Parekh and Nikolas Vitsakis and Alessandro Suglia and Ioannis Konstas},
  year={2024},
  eprint={2407.03967},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.03967},
}
```
