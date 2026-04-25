# QuPath SAM extension



A [QuPath](https://qupath.github.io) extension for running [Segment Anything Model (SAM)](https://segment-anything.com/)–style segmentation from the viewer.

## Requirements

- [QuPath](https://qupath.github.io) (version compatible with this extension build).
- A **SAM HTTP API server** on your network (usually on the same PC as QuPath). The extension defaults to something like `http://localhost:8000/sam/` in preferences.

This repo includes a **conda environment** (`samapi-env/`) that matches Sai’s working server setup on **Linux with an NVIDIA GPU (CUDA 12.x)**.
conda create -n samapi -y python=3.12
conda activate samapi
python -m pip install -U pip
python -m pip install git+https://github.com/ksugar/samapi.git
---

## 1. SAM API server (conda, easy path)

### What you need installed first

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda  
- **Git** (required: several packages install from Git URLs)  
- **Linux x86_64 + NVIDIA GPU** with a recent driver (this lock file uses CUDA 12 PyTorch wheels)

### Create the environment

From the folder that contains `samapi-env` (the same folder as this `README.md`):

```bash
cd samapi-env
conda env create -f environment.yml
conda activate samapi
```

That creates an environment named **`samapi`** with Python 3.12 and all Python dependencies from `requirements-samapi.txt` (frozen from `/home/sai/conda_envs/samapi`).

### Start the API

With `samapi` activated:

```bash
uvicorn samapi.main:app --workers 2
```

The server listens on **http://127.0.0.1:8000** by default. The QuPath extension expects the SAM base path **`http://localhost:8000/sam/`** (include the `/sam/` part in the extension dialog if your build expects it).

To allow another computer on the LAN to connect:

```bash
uvicorn samapi.main:app --workers 2 --host 0.0.0.0
```

### SAM3 (optional)

SAM3 weights are gated on Hugging Face. If you use SAM3, request access to the Meta SAM3 model, install the Hugging Face CLI, then run `hf auth login` in the same environment. If you skip this, the server can still run for SAM / SAM2 / MobileSAM.

### macOS / Windows / CPU-only

The bundled `environment.yml` + `requirements-samapi.txt` is aimed at **Linux + CUDA**. On other platforms, create a fresh `python=3.12` conda env, install **PyTorch** the way [PyTorch](https://pytorch.org) recommends for your OS (CUDA, CPU, or MPS on Apple Silicon), then install **`samapi`** with pip from the project’s own install instructions so dependencies match your machine.

---

## 2. QuPath extension (JAR)

1. Build the JAR (see below) or use the JAR file Sai sends you.  
2. In QuPath: install the extension JAR (drag-and-drop onto QuPath, or **Extensions → Manage extensions**, depending on your QuPath version), then restart QuPath.  
3. With the SAM server running, open **Extensions → SAM** and set the **Server** URL to match (e.g. `http://localhost:8000/sam/`).

## Build extension JAR from source

```bash
./gradlew shadowJar
```

The installable JAR is produced under `build/libs/`.

## License

This project is licensed under the GNU General Public License v3.0 — see the `LICENSE` file. Pre-trained model weights you download elsewhere may have their own terms.

## Third-party components

Segmentation features depend on model architectures and weights from the broader SAM ecosystem (e.g. Meta SAM, MobileSAM, SAM2/SAM3 where supported). Cite the relevant model papers when you publish work that uses them.
