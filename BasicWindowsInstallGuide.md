I’ll quickly check what `helpers.util` imports so the Windows GPU setup includes only the true runtime packages, then I’ll give you the updated minimal PowerShell commands (Conda + CUDA PyTorch + env vars + run).

Read [](file:///Users/arend/Developer/Research/graphto3d/model/__init__.py)

Read [](file:///Users/arend/Developer/Research/graphto3d/helpers/util.py)

Read [](file:///Users/arend/Developer/Research/graphto3d/helpers/util.py#177-177), lines 177 to 177

**Windows (GPU) bare-minimum setup to run the wrapper**

**Prereqs (one-time)**
- Install NVIDIA driver (so `nvidia-smi` works).
- Install Miniconda (or Anaconda) + Git.

**1) Get code + create env + install only runtime deps (PowerShell)**
```powershell
git clone https://github.com/he-dhamo/graphto3d.git
cd graphto3d

conda create -n g2s_env python=3.9 -y
conda activate g2s_env

python -m pip install --upgrade pip
pip install numpy fastapi uvicorn pydantic
pip install torch --index-url https://download.pytorch.org/whl/cu121

$env:PYTHONPATH = (Get-Location).Path
```

**2) Download pretrained checkpoints (PowerShell equivalent of the bash script)**
```powershell
cd experiments
curl -L -o final_checkpoints.zip http://campar.in.tum.de/files/graphto3d/final_checkpoints.zip
Expand-Archive -Force final_checkpoints.zip .

cd ..
```

**3) Ensure `_scene_` exists in your class vocab (only if checkpoint loading complains)**
If you get a size-mismatch error when loading weights, add `_scene_` as the first line of `GT\classes.txt`:
```powershell
$path = "GT\classes.txt"
$first = (Get-Content $path -TotalCount 1)
if ($first -ne "_scene_") { Set-Content $path ("_scene_`n" + (Get-Content $path -Raw)) }
```

**4) Point env vars at the checkpoint + vocab, then run (GPU)**
Pick the folder that contains `args.json` and a `checkpoint\` directory (commonly `experiments\final_checkpoints\shared`):

```powershell
$env:G2S_EXP_DIR = (Resolve-Path ".\experiments\final_checkpoints\shared").Path
$env:G2S_EPOCH = "100"
$env:G2S_CLASSES_PATH = (Resolve-Path ".\GT\classes.txt").Path
$env:G2S_RELATIONSHIPS_PATH = (Resolve-Path ".\GT\relationships.txt").Path
$env:G2S_DEVICE = "cuda"
```

Run the API server:
```powershell
python .\ContextAwareContentWrapper.py
```

Or run your local test (overrides the bad default because you set `G2S_EXP_DIR` above):
```powershell
python .\test_wrapper.py
```

**Important note**
- Your current code path effectively requires CUDA: `batch_torch_denormalize_box_params` in the project uses `.cuda()` unconditionally, so make sure `torch.cuda.is_available()` is true (otherwise it can crash even if you set `G2S_DEVICE=cpu`). If you want true CPU support later, I can tell you the minimal code change.