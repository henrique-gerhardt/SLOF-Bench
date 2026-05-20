from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


CONTRACT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = CONTRACT_ROOT.parent
RESULTS_DIR = CONTRACT_ROOT / "results"
OUTPUTS_DIR = CONTRACT_ROOT / "outputs"
RAW_LOGS_DIR = RESULTS_DIR / "raw_logs"
DENSITY_MASK_PATH = WORKSPACE_ROOT / "distortiondensity.npy"
DEFAULT_DATASET_ENV_VARS = ("FLOW360_ROOT", "BENCHMARK_DATASET_ROOT")
DEFAULT_CHECKPOINT_ENV_VARS = ("SLOF_CHECKPOINT", "BENCHMARK_CHECKPOINT")


def ensure_contract_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            list(cmd),
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
        ).strip()
    except Exception:
        return None


def load_contract_configs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    manifest = load_yaml(CONTRACT_ROOT / "manifest.yaml")
    datasets = load_yaml(CONTRACT_ROOT / "config" / "datasets.yaml")
    runtime = load_yaml(CONTRACT_ROOT / "config" / "runtime.yaml")
    experiment = load_yaml(CONTRACT_ROOT / "config" / "experiment.yaml")
    return manifest, datasets, runtime, experiment


def default_metadata_payload(scenario: str) -> Dict[str, Any]:
    return {
        "method_name": "slof",
        "scenario": scenario,
        "status": "pending",
        "notes": "metadata placeholder created during preflight.",
    }


def default_quality_payload() -> Dict[str, Any]:
    return {
        "epe_global": None,
        "ae_global": None,
        "eped_global": None,
        "aed_global": None,
        "epe_polar": None,
        "epe_equatorial": None,
        "epe_by_latitude": None,
        "eped_by_latitude": None,
        "valid_pixels_ratio": None,
        "notes": "quality metrics placeholder.",
    }


def default_efficiency_payload() -> Dict[str, Any]:
    return {
        "parameters": None,
        "checkpoint_size_mb": None,
        "flops_g": None,
        "latency_mean_ms": None,
        "latency_median_ms": None,
        "latency_p95_ms": None,
        "max_gpu_memory_mb": None,
        "fps": None,
        "notes": "efficiency metrics placeholder.",
    }


def ensure_json(path: Path, payload: Dict[str, Any]) -> None:
    if not path.exists():
        write_json(path, payload)


def merge_json(path: Path, payload: Dict[str, Any]) -> None:
    current = read_json(path) if path.exists() else {}
    current.update(payload)
    write_json(path, current)


def resolve_dataset_root(dataset_cfg: Dict[str, Any]) -> Path:
    candidates: List[Path] = []
    for env_var in DEFAULT_DATASET_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    root = dataset_cfg.get("root")
    if root:
        candidates.append(Path(root))

    reference = dataset_cfg.get("local_reference_root")
    if reference:
        candidates.append(Path(reference))

    checked: List[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "FLOW360 dataset root not found. Checked: " + ", ".join(checked)
    )


def resolve_checkpoint_path(experiment: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    candidates: List[Path] = []
    for env_var in DEFAULT_CHECKPOINT_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    checkpoint = experiment.get("checkpoint")
    if checkpoint:
        candidates.append(Path(checkpoint))

    default_checkpoint = manifest.get("checkpoints", {}).get("default")
    if default_checkpoint:
        candidates.append(Path(default_checkpoint))

    checked: List[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
        if candidate.is_absolute() and str(candidate).startswith("/app/"):
            local_candidate = WORKSPACE_ROOT / candidate.relative_to("/app")
            checked.append(str(local_candidate))
            if local_candidate.exists():
                return local_candidate

    raise FileNotFoundError(
        "Checkpoint not found. Checked: " + ", ".join(checked)
    )


def detect_checkpoint_variant(checkpoint_path: Path, experiment: Dict[str, Any]) -> str:
    explicit = experiment.get("checkpoint_variant")
    if explicit:
        return str(explicit).lower()

    stem = checkpoint_path.stem.lower()
    for known in (
        "singlerotation",
        "switchrotation",
        "doublerotation",
        "raftfinetune",
        "raft",
        "ktn",
    ):
        if known in stem:
            return known
    return stem


def require_cuda(runtime: Dict[str, Any]) -> torch.device:
    preferred = str(runtime.get("device", "cuda")).lower()
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by runtime.yaml but is not available.")
    if preferred.startswith("cuda"):
        return torch.device("cuda:0")
    return torch.device("cpu")


def maprange(
    x: torch.Tensor,
    minfrom: Union[torch.Tensor, float] = -1024,
    maxfrom: Union[torch.Tensor, float] = 1024,
    minto: float = 0,
    maxto: float = 1,
) -> torch.Tensor:
    return minto + ((maxto - minto) * (x - minfrom)) / (maxfrom - minfrom)


def load_density_mask(device: torch.device) -> torch.Tensor:
    dist = 1.0 - np.load(DENSITY_MASK_PATH)
    dist = 0.500 + ((1.000 - 0.500) * (dist - dist.min())) / (dist.max() - dist.min())
    return torch.from_numpy(dist).float().unsqueeze(0).to(device)


def angular_error_map(
    flow_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    ugt = flow_gt.select(1, 0)
    vgt = flow_gt.select(1, 1)
    u = flow_pred.select(1, 0)
    v = flow_pred.select(1, 1)

    ugt = ugt / (ugt**2 + vgt**2 + epsilon).sqrt()
    vgt = vgt / (ugt**2 + vgt**2 + epsilon).sqrt()
    u = u / (u**2 + v**2 + epsilon).sqrt()
    v = v / (u**2 + v**2 + epsilon).sqrt()

    var = (ugt * u + v * vgt + 1) / (
        (u**2 + v**2 + 1).sqrt() * (ugt**2 + vgt**2 + 1).sqrt()
    )
    var = maprange(var, minfrom=var.min(), maxfrom=var.max(), minto=-1, maxto=1)
    return torch.acos(var)


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        self.total += float(values.sum().item())
        self.count += int(values.numel())

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count


class Flow360PairDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        split: str,
        resize: Optional[Tuple[int, int]],
        frame_dir: str,
        forward_flow_dir: str,
        frame_suffix: str,
        flow_suffix: str,
    ) -> None:
        self.root_path = root_path
        self.split = split
        self.resize = resize
        self.frame_dir = frame_dir
        self.forward_flow_dir = forward_flow_dir
        self.frame_suffix = frame_suffix
        self.flow_suffix = flow_suffix
        self.to_tensor = ToTensor()

        split_root = self.root_path / self.split
        self.scene_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        self.frame_paths: List[Path] = []
        for scene_dir in self.scene_dirs:
            frames = sorted((scene_dir / self.frame_dir).glob(f"*{self.frame_suffix}"))
            self.frame_paths.extend(frames[:-1])

    def __len__(self) -> int:
        return len(self.frame_paths)

    def _transform_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.resize:
            image = image.resize(self.resize[::-1])
        return self.to_tensor(image)

    def _transform_flow(self, flow: np.ndarray) -> torch.Tensor:
        flow = flow.astype(np.float32)
        if self.resize:
            height, width, _ = flow.shape
            flow[:, :, 0] = flow[:, :, 0] / width
            flow[:, :, 1] = flow[:, :, 1] / height
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            flow_tensor = F.interpolate(flow_tensor, self.resize)[0]
            flow_tensor[0] = flow_tensor[0] * self.resize[1]
            flow_tensor[1] = flow_tensor[1] * self.resize[0]
            return flow_tensor.float()
        return torch.from_numpy(flow).permute(2, 0, 1).float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame1_path = self.frame_paths[index]
        frame2_path = frame1_path.parent / f"{int(frame1_path.stem) + 1:04d}{frame1_path.suffix}"
        flow_path = Path(
            frame1_path.as_posix()
            .replace(f"/{self.frame_dir}/", f"/{self.forward_flow_dir}/")
            .replace(self.frame_suffix, self.flow_suffix)
        )

        frame1 = self._transform_image(Image.open(frame1_path))
        frame2 = self._transform_image(Image.open(frame2_path))
        flow = self._transform_flow(-np.load(flow_path))
        valid = ((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float()

        return {
            "frame1": frame1,
            "frame2": frame2,
            "fflow": flow,
            "valid": valid,
            "frame1_path": str(frame1_path),
            "frame2_path": str(frame2_path),
            "flow_path": str(flow_path),
            "scene": frame1_path.parent.parent.name,
            "frame_index": frame1_path.stem,
        }


def build_dataloader(
    dataset_root: Path,
    dataset_cfg: Dict[str, Any],
    split: str,
    resize: Optional[Tuple[int, int]],
    batch_size: int,
    num_workers: int,
) -> Tuple[Flow360PairDataset, DataLoader]:
    dataset = Flow360PairDataset(
        root_path=dataset_root,
        split=split,
        resize=resize,
        frame_dir=str(dataset_cfg.get("frame_dir", "frames")),
        forward_flow_dir=str(dataset_cfg.get("forward_flow_dir", "fflows")),
        frame_suffix=str(dataset_cfg.get("frame_suffix", ".png")),
        flow_suffix=str(dataset_cfg.get("flow_suffix", ".npy")),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def build_ktnized_raft(device_index: int) -> torch.nn.Module:
    from KernelTransformerNetwork.KernelTransformer import KTNLayer
    from RAFT import load_raft

    raft = load_raft.load(
        path_root=(WORKSPACE_ROOT / "RAFT").as_posix(),
        data_parallel=False,
        simsiam=True,
        load=False,
        DEVICE_IDS=[device_index],
    ).train(False)

    raft.fnet.conv1 = KTNLayer.KTNConv(
        raft.fnet.conv1.weight.data.cuda(),
        raft.fnet.conv1.bias.data.cuda(),
        sphereH=320,
        imgW=640,
        tied_weights=20,
        output_shape=(160, 320),
    ).cuda()
    raft.fnet.layer1[0].conv1 = KTNLayer.KTNConv(
        raft.fnet.layer1[0].conv1.weight.data.cuda(),
        raft.fnet.layer1[0].conv1.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer1[0].conv2 = KTNLayer.KTNConv(
        raft.fnet.layer1[0].conv2.weight.data.cuda(),
        raft.fnet.layer1[0].conv2.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer1[1].conv1 = KTNLayer.KTNConv(
        raft.fnet.layer1[1].conv1.weight.data.cuda(),
        raft.fnet.layer1[1].conv1.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer1[1].conv2 = KTNLayer.KTNConv(
        raft.fnet.layer1[1].conv2.weight.data.cuda(),
        raft.fnet.layer1[1].conv2.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer2[0].conv1 = KTNLayer.KTNConv(
        raft.fnet.layer2[0].conv1.weight.data.cuda(),
        raft.fnet.layer2[0].conv1.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
        output_shape=(80, 160),
    ).cuda()
    raft.fnet.layer2[0].conv2 = KTNLayer.KTNConv(
        raft.fnet.layer2[0].conv2.weight.data.cuda(),
        raft.fnet.layer2[0].conv2.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer2[1].conv1 = KTNLayer.KTNConv(
        raft.fnet.layer2[1].conv1.weight.data.cuda(),
        raft.fnet.layer2[1].conv1.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer2[1].conv2 = KTNLayer.KTNConv(
        raft.fnet.layer2[1].conv2.weight.data.cuda(),
        raft.fnet.layer2[1].conv2.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
    ).cuda()
    raft.fnet.layer3[0].conv1 = KTNLayer.KTNConv(
        raft.fnet.layer3[0].conv1.weight.data.cuda(),
        raft.fnet.layer3[0].conv1.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
        output_shape=(40, 80),
    ).cuda()
    raft.fnet.layer3[0].conv2 = KTNLayer.KTNConv(
        raft.fnet.layer3[0].conv2.weight.data.cuda(),
        raft.fnet.layer3[0].conv2.bias.data.cuda(),
        sphereH=40,
        imgW=80,
        tied_weights=5,
    ).cuda()
    raft.fnet.layer3[1].conv1 = KTNLayer.KTNConv(
        raft.fnet.layer3[1].conv1.weight.data.cuda(),
        raft.fnet.layer3[1].conv1.bias.data.cuda(),
        sphereH=40,
        imgW=80,
        tied_weights=5,
    ).cuda()
    raft.fnet.layer3[1].conv2 = KTNLayer.KTNConv(
        raft.fnet.layer3[1].conv2.weight.data.cuda(),
        raft.fnet.layer3[1].conv2.bias.data.cuda(),
        sphereH=40,
        imgW=80,
        tied_weights=5,
    ).cuda()

    raft.cnet.conv1 = KTNLayer.KTNConv(
        raft.cnet.conv1.weight.data.cuda(),
        raft.cnet.conv1.bias.data.cuda(),
        sphereH=320,
        imgW=640,
        tied_weights=20,
        output_shape=(160, 320),
    ).cuda()
    raft.cnet.layer1[0].conv1 = KTNLayer.KTNConv(
        raft.cnet.layer1[0].conv1.weight.data.cuda(),
        raft.cnet.layer1[0].conv1.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer1[0].conv2 = KTNLayer.KTNConv(
        raft.cnet.layer1[0].conv2.weight.data.cuda(),
        raft.cnet.layer1[0].conv2.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer1[1].conv1 = KTNLayer.KTNConv(
        raft.cnet.layer1[1].conv1.weight.data.cuda(),
        raft.cnet.layer1[1].conv1.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer1[1].conv2 = KTNLayer.KTNConv(
        raft.cnet.layer1[1].conv2.weight.data.cuda(),
        raft.cnet.layer1[1].conv2.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer2[0].conv1 = KTNLayer.KTNConv(
        raft.cnet.layer2[0].conv1.weight.data.cuda(),
        raft.cnet.layer2[0].conv1.bias.data.cuda(),
        sphereH=160,
        imgW=320,
        tied_weights=20,
        output_shape=(80, 160),
    ).cuda()
    raft.cnet.layer2[0].conv2 = KTNLayer.KTNConv(
        raft.cnet.layer2[0].conv2.weight.data.cuda(),
        raft.cnet.layer2[0].conv2.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer2[1].conv1 = KTNLayer.KTNConv(
        raft.cnet.layer2[1].conv1.weight.data.cuda(),
        raft.cnet.layer2[1].conv1.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer2[1].conv2 = KTNLayer.KTNConv(
        raft.cnet.layer2[1].conv2.weight.data.cuda(),
        raft.cnet.layer2[1].conv2.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
    ).cuda()
    raft.cnet.layer3[0].conv1 = KTNLayer.KTNConv(
        raft.cnet.layer3[0].conv1.weight.data.cuda(),
        raft.cnet.layer3[0].conv1.bias.data.cuda(),
        sphereH=80,
        imgW=160,
        tied_weights=20,
        output_shape=(40, 80),
    ).cuda()
    raft.cnet.layer3[0].conv2 = KTNLayer.KTNConv(
        raft.cnet.layer3[0].conv2.weight.data.cuda(),
        raft.cnet.layer3[0].conv2.bias.data.cuda(),
        sphereH=40,
        imgW=80,
        tied_weights=5,
    ).cuda()
    raft.cnet.layer3[1].conv1 = KTNLayer.KTNConv(
        raft.cnet.layer3[1].conv1.weight.data.cuda(),
        raft.cnet.layer3[1].conv1.bias.data.cuda(),
        sphereH=40,
        imgW=80,
        tied_weights=5,
    ).cuda()
    raft.cnet.layer3[1].conv2 = KTNLayer.KTNConv(
        raft.cnet.layer3[1].conv2.weight.data.cuda(),
        raft.cnet.layer3[1].conv2.bias.data.cuda(),
        sphereH=40,
        imgW=80,
        tied_weights=5,
    ).cuda()
    return raft.cuda()


def load_encoder_for_checkpoint(
    checkpoint_path: Path,
    checkpoint_variant: str,
    device: torch.device,
) -> torch.nn.Module:
    from RAFT import load_raft
    from simsiam import Siam360

    device_index = 0 if device.index is None else device.index
    if checkpoint_variant == "ktn":
        encoder_backbone = build_ktnized_raft(device_index=device_index).train(False)
        model = Siam360(encoder_backbone, finetune=True)
    else:
        finetune = checkpoint_variant in {"raft", "raftfinetune"}
        encoder_backbone = load_raft.load(
            path_root=(WORKSPACE_ROOT / "RAFT").as_posix(),
            data_parallel=False,
            simsiam=True,
            load=False,
            DEVICE_IDS=[device_index],
        ).train(False)
        model = Siam360(encoder_backbone, finetune=finetune)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    encoder = model.encoder.eval()
    del model
    torch.cuda.empty_cache()
    return encoder


def count_encoder_parameters(encoder: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in encoder.parameters())


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return sorted(values)[index]


def tensor_shape_list(tensor: torch.Tensor) -> List[int]:
    return [int(item) for item in tensor.shape]


def collect_environment_snapshot() -> Dict[str, Any]:
    torch_version = None
    torch_cuda = None
    cudnn_version = None
    try:
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        pass

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda,
        "cudnn_version": cudnn_version,
        "gpu_name": safe_cmd(
            ["bash", "-lc", "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"]
        ),
        "driver_version": safe_cmd(
            ["bash", "-lc", "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1"]
        ),
        "cuda_runtime_reported": safe_cmd(
            [
                "bash",
                "-lc",
                "nvidia-smi | sed -n 's/.*CUDA Version: \\([0-9.]*\\).*/\\1/p' | head -n 1",
            ]
        ),
        "git_commit": safe_cmd(["git", "rev-parse", "HEAD"], cwd=WORKSPACE_ROOT),
    }
