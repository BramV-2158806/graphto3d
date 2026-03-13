import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from fastapi import FastAPI, HTTPException
except Exception:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:  # Minimal no-op fallback for non-API usage.
        def __init__(self, *args, **kwargs):
            pass

        def on_event(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def get(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def post(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

try:
    from pydantic import BaseModel, Field
except Exception:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self):
            return self.__dict__

        def json(self, indent=None):
            return json.dumps(self.__dict__, indent=indent)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

from helpers.util import batch_torch_denormalize_box_params, normalize_box_params
from model.VAE import VAE


@dataclass
class NodeBox:
    label: str
    size: np.ndarray
    center: np.ndarray
    yaw_rad: float = 0.0


@dataclass
class Addition:
    object_name: str
    target_name: str
    relation: str
    function_name: str


# Default label-to-class normalization map.
#
# Why this exists:
# - Unity scene labels are often project-specific (for example FrontHub, Storage Cabinet),
#   while GraphTo3D expects labels from its training vocabulary (for example chair, cabinet).
# - This map bridges those naming systems.
#
# How to read it:
# - key: raw label expected from scene_graph nodes (case-insensitive)
# - value: canonical class name that must exist in classes.txt
#
# How to customize:
# - Keep values aligned with your GraphTo3D vocabulary.
# - Prefer broad stable fallback classes for synthetic UI objects (for example text panels -> picture/chair).
# - You can override these defaults per API request via class_alias_map.
DEFAULT_CLASS_ALIAS_MAP: Dict[str, str] = {
    # Room structure / fixtures
    "brickwall": "wall",
    "brickwall (2)": "wall",
    "brickwall (3)": "wall",
    "brick wall (4)": "wall",
    "concretefloor": "floor",
    "garage port": "door",
    "window (1)": "window",
    "window (2)": "window",
    "storage cabinet": "cabinet",
    "boiler": "heater",

    # Bicycle aggregate / unknown generated ids
    "carbonframebike_0": "object",
    "zb_vr_299": "object",

    # Bicycle frame and cockpit
    "frame": "frame",
    "frontfork": "object",
    "frontfork(2)": "object",
    "stem": "object",
    "handlebars": "object",
    "leftgrip": "object",
    "rightgrip": "object",
    "seatpost": "pipe",
    "saddle": "object",

    # Wheels and tire system
    "frontwheel": "object",
    "rearwheel": "object",
    "fronttire": "object",
    "reartire": "object",
    "frontrim": "object",
    "rearrim": "object",
    "frontspokes": "object",
    "rearspokes": "object",
    "fronthub": "object",
    "rearhub": "object",
    "frontright hubnut": "object",
    "rearrighthubnut": "object",
    "rearlefthubnut": "object",

    # Drivetrain and brake details
    "chain": "object",
    "chainrings": "object",
    "rightcrankandpedal": "object",
    "leftcrankandpedal": "object",
    "rearcasette": "object",
    "rearderailleur": "object",
    "rearrightchainstay": "pipe",
    "rearleftchainstay": "pipe",
    "frontdiskbrakerotor": "object",
    "reardiskbrakerotor": "object",
    "frontcalliper": "object",
    "rearcalliper": "object",
}


# Default relation normalization map.
#
# Why this exists:
# - Architect and Unity relations (for example InFrontOf, NextTo, OnTop) need to be
#   converted into predicate names used by relationships.txt (for example front, close by, standing on).
#
# How to read it:
# - key: incoming relation string (case-insensitive)
# - value: canonical predicate that must exist in relationships.txt
#
# How to customize:
# - Add project-specific relation spellings and synonyms here.
# - If a relation is missing, the edge may be skipped during graph construction.
# - You can override these defaults per API request via relation_alias_map.
DEFAULT_RELATION_ALIAS_MAP: Dict[str, str] = {
    "infrontof": "front",
    "front": "front",
    "behind": "behind",

    "leftof": "left",
    "left": "left",
    "rightof": "right",
    "right": "right",

    "nextto": "close by",
    "next to": "close by",

    "inside": "inside",

    # Keep edges instead of dropping them; semantic direction is imperfect.
    "contains": "close by",

    "ontop": "standing on",
    "on top": "standing on",
    "under": "lower than",
}


class PredictRequest(BaseModel):
    scene_graph: Dict[str, Any]
    architect_response: Dict[str, Any]
    class_alias_map: Dict[str, str] = Field(default_factory=dict)
    relation_alias_map: Dict[str, str] = Field(default_factory=dict)
    default_added_class: str = "picture"


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    skipped_additions: List[Dict[str, str]]


class ContextAwareContentWrapper:
    def __init__(
        self,
        exp_dir: str,
        epoch: str,
        classes_txt_path: str,
        relationships_txt_path: str,
        class_alias_map: Optional[Dict[str, str]] = None,
        relation_alias_map: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        add_scene_node: bool = True,
    ):
        """Initialize the scene-graph-to-bounding-box wrapper.

        Parameters:
            exp_dir: Path to the GraphTo3D experiment directory containing args.json and checkpoint/.
                Example: "./experiments/final_checkpoints/shared"
            epoch: Checkpoint epoch identifier used by model.load_networks.
                Example: "100"
            classes_txt_path: Path to classes vocabulary file (one class per line).
                Example: "./GT/classes.txt"
            relationships_txt_path: Path to relationships vocabulary file.
                Example: "./GT/relationships.txt"
            class_alias_map: Optional Unity-label to GraphTo3D-class mapping overrides.
                Example: {"FrontHub": "chair", "Storage Cabinet": "cabinet"}
            relation_alias_map: Optional Unity relation to GraphTo3D relation mapping overrides.
                Example: {"InFrontOf": "front", "NextTo": "close by"}
            device: Preferred torch device. Falls back to CPU if CUDA is unavailable.
                Example: "cuda" or "cpu"
            add_scene_node: If True, appends _scene_ node and in-scene edges (matches training pattern).
                Example: True

        Notes:
            - This loads the VAE model weights immediately.
            - Alias maps are merged with defaults and normalized to lowercase.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.add_scene_node = add_scene_node

        self.vocab = self._load_vocab(classes_txt_path, relationships_txt_path)

        args_path = os.path.join(exp_dir, "args.json")
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"Could not find model args at: {args_path}")

        with open(args_path, "r") as f:
            model_args = json.load(f)

        self.model = VAE(
            type=model_args["network_type"],
            vocab=self.vocab,
            replace_latent=model_args.get("replace_latent"),
            with_changes=model_args.get("with_changes"),
            residual=model_args["residual"],
            gconv_pooling=model_args["pooling"],
            with_angles=model_args["with_angles"],
        )
        self.model.load_networks(exp=exp_dir, epoch=epoch, map_location=self.device)
        self.model = self.model.to(self.device).eval()

        merged_class_alias = dict(DEFAULT_CLASS_ALIAS_MAP)
        if class_alias_map:
            merged_class_alias.update(class_alias_map)
        self.class_alias_map = {k.lower(): v.lower() for k, v in merged_class_alias.items()}

        merged_relation_alias = dict(DEFAULT_RELATION_ALIAS_MAP)
        if relation_alias_map:
            merged_relation_alias.update(relation_alias_map)
        self.relation_alias_map = {k.lower(): v.lower() for k, v in merged_relation_alias.items()}

    def _load_vocab(self, classes_txt_path: str, rels_txt_path: str) -> Dict[str, Any]:
        """Load class and relationship vocab files into forward and reverse lookup structures.

        Parameters:
            classes_txt_path: Path to classes.txt.
                Example: "./GT/classes.txt"
            rels_txt_path: Path to relationships.txt.
                Example: "./GT/relationships.txt"

        Returns:
            Dictionary with:
                - object_idx_to_name: list[str]
                - pred_idx_to_name: list[str]
                - object_name_to_idx: dict[str, int]
                - pred_name_to_idx: dict[str, int]
        """
        if not os.path.exists(classes_txt_path):
            raise FileNotFoundError(f"Could not find classes file at: {classes_txt_path}")
        if not os.path.exists(rels_txt_path):
            raise FileNotFoundError(f"Could not find relationships file at: {rels_txt_path}")

        with open(classes_txt_path, "r") as f:
            obj_idx_to_name = f.readlines()
        with open(rels_txt_path, "r") as f:
            pred_idx_to_name = f.readlines()

        object_name_to_idx: Dict[str, int] = {}
        for i, name in enumerate(obj_idx_to_name):
            object_name_to_idx[name.strip().lower()] = i

        pred_name_to_idx: Dict[str, int] = {}
        for i, name in enumerate(pred_idx_to_name):
            pred_name_to_idx[name.strip().lower()] = i

        return {
            "object_idx_to_name": obj_idx_to_name,
            "pred_idx_to_name": pred_idx_to_name,
            "object_name_to_idx": object_name_to_idx,
            "pred_name_to_idx": pred_name_to_idx,
        }

    def parse_scene_graph(self, scene_graph: Dict[str, Any]) -> Dict[str, NodeBox]:
        """Parse Unity scene graph JSON into a label->NodeBox mapping.

        Parameters:
            scene_graph: Raw Unity scene graph dictionary with keys like nodes/edges.
                Example: {"nodes": [{"label": "FrontHub", "worldBounds": {...}}], "edges": [...]}

        Returns:
            Mapping of node label to NodeBox containing size/center in model axis convention.

        Notes:
            - Unity axes are converted from xyz to xzy for model compatibility.
            - Nodes missing label/center/size are skipped.
        """
        nodes: Dict[str, NodeBox] = {}
        for n in scene_graph.get("nodes", []):
            label = str(n.get("label", "")).strip()
            wb = n.get("worldBounds", {})
            center = wb.get("center", {})
            size = wb.get("size", {})

            if not label or not center or not size:
                continue

            # Model uses [w, l, h, cx, cy, cz]. We map Unity xyz -> xzy.
            node = NodeBox(
                label=label,
                size=np.array(
                    [
                        float(size.get("x", 0.1)),
                        float(size.get("z", 0.1)),
                        float(size.get("y", 0.1)),
                    ],
                    dtype=np.float32,
                ),
                center=np.array(
                    [
                        float(center.get("x", 0.0)),
                        float(center.get("z", 0.0)),
                        float(center.get("y", 0.0)),
                    ],
                    dtype=np.float32,
                ),
                yaw_rad=0.0,
            )
            nodes[label] = node
        return nodes

    def parse_architect_additions(self, architect_response: Dict[str, Any]) -> List[Addition]:
        """Parse architect tool calls into a normalized list of additions.

        Parameters:
            architect_response: Architect response dictionary containing toolCalls.
                Example: {"toolCalls": [{"functionName": "CreateText", "arguments": {...}}]}

        Returns:
            List of Addition entries containing new object name, target object name, and semantic relation.

        Notes:
            - Only CreateText and CreateTextTo2D are consumed.
            - Incomplete entries are skipped.
        """
        out: List[Addition] = []
        for c in architect_response.get("toolCalls", []):
            function_name = str(c.get("functionName", "")).strip()
            if function_name not in ("CreateText", "CreateTextTo2D"):
                continue

            args = c.get("arguments", {})
            obj_name = str(args.get("object_name", "")).strip()
            target_name = str(args.get("target_object_name", "")).strip()
            relation = str(args.get("semantic_relation", "")).strip()

            if not obj_name or not target_name or not relation:
                continue

            out.append(
                Addition(
                    object_name=obj_name,
                    target_name=target_name,
                    relation=relation,
                    function_name=function_name,
                )
            )
        return out

    def _canonical_class(self, unity_label: str) -> Optional[str]:
        """Map a Unity object label to a canonical GraphTo3D class name.

        Parameters:
            unity_label: Original Unity label.
                Example: "Storage Cabinet"

        Returns:
            Canonical class name if found, otherwise None.
            Example return: "cabinet"
        """
        key = unity_label.lower()
        if key in self.class_alias_map:
            mapped = self.class_alias_map[key]
            if mapped in self.vocab["object_name_to_idx"]:
                return mapped

        if key in self.vocab["object_name_to_idx"]:
            return key
        return None

    def _canonical_relation(self, relation: str) -> Optional[str]:
        """Map an input relation to a canonical relationship in GraphTo3D vocabulary.

        Parameters:
            relation: Relation string from Unity/architect.
                Example: "InFrontOf"

        Returns:
            Canonical relationship if found, otherwise None.
            Example return: "front"
        """
        key = relation.lower()
        if key in self.relation_alias_map:
            key = self.relation_alias_map[key]

        if key in self.vocab["pred_name_to_idx"]:
            return key
        return None

    def _box7_metric(self, node: NodeBox) -> np.ndarray:
        """Convert NodeBox to 7D metric box format expected by preprocessing.

        Parameters:
            node: Parsed node container.
                Example: NodeBox(label="FrontHub", size=[0.2, 0.2, 0.1], center=[0.1, -4.0, 0.4], yaw_rad=0.0)

        Returns:
            np.ndarray of shape [7] = [w, l, h, cx, cy, cz, yaw_rad].
        """
        return np.array(
            [
                node.size[0],
                node.size[1],
                node.size[2],
                node.center[0],
                node.center[1],
                node.center[2],
                node.yaw_rad,
            ],
            dtype=np.float32,
        )

    def _make_initial_added_box(self, target_box7: np.ndarray, relation: str) -> np.ndarray:
        """Create an initial guess box for a newly added object around a target object.

        Parameters:
            target_box7: Target object's metric box [w, l, h, cx, cy, cz, yaw].
                Example: np.array([0.6, 0.6, 0.1, 0.3, -4.1, 0.4, 0.0], dtype=np.float32)
            relation: Spatial relation used to offset the initial center.
                Example: "InFrontOf"

        Returns:
            New 7D box guess near the target object.

        Notes:
            - This is a heuristic seed; the network predicts the final placement.
        """
        box = target_box7.copy()
        w, l, h = box[0], box[1], box[2]
        cx, cy, cz = box[3], box[4], box[5]

        delta = max(float(w), float(l), float(h)) * 0.6 + 0.05
        rel = relation.lower()

        if rel in ("infrontof", "front"):
            cy -= delta
        elif rel in ("behind",):
            cy += delta
        elif rel in ("leftof", "left"):
            cx -= delta
        elif rel in ("rightof", "right"):
            cx += delta
        elif rel in ("ontop", "standing on", "supported by"):
            cz += h * 0.5 + 0.05
        elif rel in ("under",):
            cz -= h * 0.5 + 0.05

        box[3], box[4], box[5] = cx, cy, cz
        return box

    def _normalize_boxes_for_model(self, boxes7: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize metric boxes and discretize angles for model encoding.

        Parameters:
            boxes7: List of metric boxes [w, l, h, cx, cy, cz, yaw_rad].
                Example: [np.array([1.0, 0.5, 0.7, 0.2, -4.0, 0.6, 0.0], dtype=np.float32)]
            labels: Label list aligned with boxes7.
                Example: ["FrontHub", "_scene_"]

        Returns:
            Tuple:
                - boxes6: np.ndarray [N, 6] normalized for the model
                - angles: np.ndarray [N] angle bin index in [0, 23]

        Notes:
            - _scene_ node is passed through as [-1, -1, -1, -1, -1, -1] with angle 0.
        """
        boxes6: List[np.ndarray] = []
        angles: List[int] = []
        bins = np.linspace(0, 2 * math.pi, 24)

        for b, label in zip(boxes7, labels):
            if label == "_scene_":
                boxes6.append(np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32))
                angles.append(0)
                continue

            bn = normalize_box_params(b.copy())
            boxes6.append(bn[:6].astype(np.float32))

            angle = float(np.mod(b[6], 2 * math.pi))
            angle_idx = int(np.digitize(angle, bins) - 1)
            angle_idx = max(0, min(angle_idx, 23))
            angles.append(angle_idx)

        return np.asarray(boxes6, dtype=np.float32), np.asarray(angles, dtype=np.int64)

    def build_model_inputs(
        self,
        scene_nodes: Dict[str, NodeBox],
        scene_edges: List[Dict[str, Any]],
        additions: List[Addition],
        default_added_class: str,
    ) -> Dict[str, Any]:
        """Build encoder/decoder tensors and metadata used by GraphTo3D manipulation inference.

        Parameters:
            scene_nodes: Parsed scene nodes mapping from parse_scene_graph.
                Example: {"FrontHub": NodeBox(...), "RearHub": NodeBox(...)}
            scene_edges: Raw edge list from scene_graph["edges"].
                Example: [{"subjectId": "FrontHub", "targetId": "Frame", "relation": "NextTo"}]
            additions: Parsed additions from parse_architect_additions.
                Example: [Addition(object_name="step1_front_remove_wheel", target_name="FrontHub", relation="InFrontOf", function_name="CreateText")]
            default_added_class: Fallback class for new labels that do not map to known vocab.
                Example: "picture"

        Returns:
            Dictionary containing torch tensors and helper metadata:
                - enc_objs, enc_triples, enc_boxes6, enc_angles
                - dec_objs, dec_triples
                - missing_nodes, manipulated_nodes
                - dec_labels, skipped_additions

        Raises:
            ValueError if no nodes/relations/additions could be mapped to a valid model input.
        """
        enc_labels: List[str] = []
        enc_objs: List[int] = []
        enc_boxes7: List[np.ndarray] = []
        label_to_enc_idx: Dict[str, int] = {}

        for label, node in scene_nodes.items():
            cls = self._canonical_class(label)
            if cls is None:
                continue
            cls_idx = self.vocab["object_name_to_idx"][cls]

            label_to_enc_idx[label] = len(enc_objs)
            enc_labels.append(label)
            enc_objs.append(cls_idx)
            enc_boxes7.append(self._box7_metric(node))

        if len(enc_objs) == 0:
            raise ValueError("No scene nodes matched known classes. Please expand class_alias_map.")

        scene_root_idx_enc: Optional[int] = None
        if self.add_scene_node and "_scene_" in self.vocab["object_name_to_idx"]:
            scene_root_idx_enc = len(enc_objs)
            enc_labels.append("_scene_")
            enc_objs.append(self.vocab["object_name_to_idx"]["_scene_"])
            enc_boxes7.append(np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32))

        enc_triples: List[List[int]] = []
        for e in scene_edges:
            s = str(e.get("subjectId", "")).strip()
            o = str(e.get("targetId", "")).strip()
            r = str(e.get("relation", "")).strip()

            if s not in label_to_enc_idx or o not in label_to_enc_idx:
                continue

            rel_name = self._canonical_relation(r)
            if rel_name is None:
                continue

            p_idx = self.vocab["pred_name_to_idx"][rel_name]
            enc_triples.append([label_to_enc_idx[s], p_idx, label_to_enc_idx[o]])

        if scene_root_idx_enc is not None:
            in_scene_idx = self.vocab["pred_name_to_idx"].get("in scene", 0)
            for idx, label in enumerate(enc_labels):
                if label == "_scene_":
                    continue
                enc_triples.append([idx, in_scene_idx, scene_root_idx_enc])

        if len(enc_triples) == 0:
            raise ValueError("No scene triples matched known relations. Please expand relation_alias_map.")

        dec_labels = list(enc_labels)
        dec_objs = list(enc_objs)
        dec_boxes7 = list(enc_boxes7)
        dec_triples = list(enc_triples)
        label_to_dec_idx = {label: i for i, label in enumerate(dec_labels)}

        missing_nodes: List[int] = []
        skipped_additions: List[Dict[str, str]] = []

        default_cls = default_added_class.lower().strip()
        if default_cls not in self.vocab["object_name_to_idx"]:
            default_cls = "picture" if "picture" in self.vocab["object_name_to_idx"] else next(iter(self.vocab["object_name_to_idx"]))
        default_cls_idx = self.vocab["object_name_to_idx"][default_cls]

        for add in additions:
            if add.target_name not in label_to_dec_idx:
                skipped_additions.append(
                    {
                        "object_name": add.object_name,
                        "reason": f"Target not found in scene graph: {add.target_name}",
                    }
                )
                continue

            if add.object_name in label_to_dec_idx:
                skipped_additions.append(
                    {
                        "object_name": add.object_name,
                        "reason": "Object already exists in scene graph",
                    }
                )
                continue

            new_class = self._canonical_class(add.object_name)
            cls_idx = default_cls_idx if new_class is None else self.vocab["object_name_to_idx"].get(new_class, default_cls_idx)

            target_idx = label_to_dec_idx[add.target_name]
            init_box = self._make_initial_added_box(dec_boxes7[target_idx], add.relation)

            new_idx = len(dec_objs)
            dec_labels.append(add.object_name)
            dec_objs.append(cls_idx)
            dec_boxes7.append(init_box)
            label_to_dec_idx[add.object_name] = new_idx

            rel_name = self._canonical_relation(add.relation)
            if rel_name is not None:
                rel_idx = self.vocab["pred_name_to_idx"][rel_name]
                dec_triples.append([new_idx, rel_idx, target_idx])
            else:
                skipped_additions.append(
                    {
                        "object_name": add.object_name,
                        "reason": f"Unknown relation: {add.relation}",
                    }
                )

            if scene_root_idx_enc is not None:
                in_scene_idx = self.vocab["pred_name_to_idx"].get("in scene", 0)
                dec_triples.append([new_idx, in_scene_idx, scene_root_idx_enc])

            missing_nodes.append(new_idx)

        if len(missing_nodes) == 0:
            raise ValueError("No valid additions to place after parsing architect_response.")

        enc_b6, enc_angles = self._normalize_boxes_for_model(enc_boxes7, enc_labels)

        inputs = {
            "enc_objs": torch.tensor(enc_objs, dtype=torch.long, device=self.device),
            "enc_triples": torch.tensor(enc_triples, dtype=torch.long, device=self.device),
            "enc_boxes6": torch.tensor(enc_b6, dtype=torch.float32, device=self.device),
            "enc_angles": torch.tensor(enc_angles, dtype=torch.long, device=self.device),
            "dec_objs": torch.tensor(dec_objs, dtype=torch.long, device=self.device),
            "dec_triples": torch.tensor(dec_triples, dtype=torch.long, device=self.device),
            "missing_nodes": missing_nodes,
            "manipulated_nodes": [],
            "dec_labels": dec_labels,
            "skipped_additions": skipped_additions,
        }
        return inputs

    @torch.no_grad()
    def predict_addition_boxes(self, model_inputs: Dict[str, Any]) -> PredictResponse:
        """Run model inference and return predicted bounding boxes for newly added nodes.

        Parameters:
            model_inputs: Output of build_model_inputs.
                Example: {"enc_objs": tensor(...), "dec_objs": tensor(...), "missing_nodes": [12, 13], ...}

        Returns:
            PredictResponse with:
                - predictions: list of boxes for nodes where keep == 0 (new/changed nodes)
                - skipped_additions: reasons for additions that could not be processed

        Output box fields:
            - box6_model_axes: w,l,h,cx,cy,cz in model axis convention
            - unity_size: x,y,z converted back for Unity scaling
            - unity_center: x,y,z converted back for Unity positioning
            - unity_yaw_deg: optional heading estimate when angle head is available
        """
        enc_objs = model_inputs["enc_objs"]
        enc_triples = model_inputs["enc_triples"]
        enc_boxes6 = model_inputs["enc_boxes6"]
        enc_angles = model_inputs["enc_angles"]

        dec_objs = model_inputs["dec_objs"]
        dec_triples = model_inputs["dec_triples"]
        missing_nodes = model_inputs["missing_nodes"]
        manipulated_nodes = model_inputs["manipulated_nodes"]
        dec_labels = model_inputs["dec_labels"]

        # The shared model does not implement VAE.encode_box; use its full encoder
        # with zero shape features when point/shape embeddings are unavailable.
        if getattr(self.model, "type_", None) == "shared":
            shape_feats = torch.zeros((enc_objs.shape[0], 128), dtype=torch.float32, device=self.device)
            z_box, _ = self.model.vae.encoder(
                enc_objs,
                enc_triples,
                enc_boxes6,
                shape_feats,
                attributes=None,
                angles_gt=enc_angles,
            )
            out, keep = self.model.vae.decoder_with_additions(
                z_box,
                dec_objs,
                dec_triples,
                attributes=None,
                missing_nodes=missing_nodes,
                manipulated_nodes=manipulated_nodes,
            )
        else:
            z_box, _ = self.model.encode_box(enc_objs, enc_triples, enc_boxes6, enc_angles, attributes=None)

            out, keep = self.model.decoder_with_changes_boxes(
                z_box,
                dec_objs,
                dec_triples,
                attributes=None,
                missing_nodes=missing_nodes,
                manipulated_nodes=manipulated_nodes,
            )

        if isinstance(out, tuple):
            boxes_pred_norm = out[0]
            angles_pred = out[1]
        else:
            boxes_pred_norm = out
            angles_pred = None

        boxes_pred_den = batch_torch_denormalize_box_params(boxes_pred_norm)

        predictions: List[Dict[str, Any]] = []
        for i in range(boxes_pred_den.shape[0]):
            if int(keep[i].item()) != 0:
                continue

            b = boxes_pred_den[i].detach().cpu().numpy().tolist()

            yaw_deg = None
            if angles_pred is not None:
                yaw_bin = int(torch.argmax(angles_pred[i]).item())
                yaw_deg = (yaw_bin / 24.0) * 360.0

            predictions.append(
                {
                    "name": dec_labels[i],
                    "box6_model_axes": {
                        "w": b[0],
                        "l": b[1],
                        "h": b[2],
                        "cx": b[3],
                        "cy": b[4],
                        "cz": b[5],
                    },
                    # Converted back to Unity axis order
                    "unity_size": {"x": b[0], "y": b[2], "z": b[1]},
                    "unity_center": {"x": b[3], "y": b[5], "z": b[4]},
                    "unity_yaw_deg": yaw_deg,
                    "decoder_index": i,
                }
            )

        return PredictResponse(
            predictions=predictions,
            skipped_additions=model_inputs["skipped_additions"],
        )


app = FastAPI(title="ContextAwareContentWrapper", version="0.1.0")
_wrapper: Optional[ContextAwareContentWrapper] = None


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    if val is None:
        return None
    v = val.strip()
    return v if v else None


def _build_wrapper() -> ContextAwareContentWrapper:
    exp_dir = _get_env("G2S_EXP_DIR")
    epoch = _get_env("G2S_EPOCH", "100")
    classes_path = _get_env("G2S_CLASSES_PATH")
    rels_path = _get_env("G2S_RELATIONSHIPS_PATH")
    device = _get_env("G2S_DEVICE", "cuda")

    missing = []
    if exp_dir is None:
        missing.append("G2S_EXP_DIR")
    if classes_path is None:
        missing.append("G2S_CLASSES_PATH")
    if rels_path is None:
        missing.append("G2S_RELATIONSHIPS_PATH")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return ContextAwareContentWrapper(
        exp_dir=exp_dir,
        epoch=epoch,
        classes_txt_path=classes_path,
        relationships_txt_path=rels_path,
        device=device,
    )


@app.on_event("startup")
def on_startup() -> None:
    global _wrapper
    _wrapper = _build_wrapper()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _wrapper is not None,
        "device": str(_wrapper.device) if _wrapper is not None else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Primary API inference entry point of the wrapper service.

    This route is the main endpoint that external clients call to get object
    placement predictions from GraphTo3D. It takes a scene graph plus architect
    additions, applies request-scoped alias overrides, builds model tensors,
    runs inference, and returns predicted boxes for new objects.

    Args:
        req: Request payload containing:
            - scene_graph: Current scene nodes and edges from the client.
            - architect_response: Tool-call output describing additions.
            - class_alias_map: Optional per-request class alias overrides.
            - relation_alias_map: Optional per-request relation alias overrides.
            - default_added_class: Fallback class for unmapped added labels.

    Returns:
        PredictResponse with:
            - predictions: Predicted placements in model and Unity axis formats.
            - skipped_additions: Additions that were skipped with reasons.

    Raises:
        HTTPException 400: Input parsed but cannot produce valid model inputs
        (for example no valid additions or unmapped graph content).
        HTTPException 500: Wrapper initialization/runtime errors.
    """
    global _wrapper
    if _wrapper is None:
        raise HTTPException(status_code=500, detail="Wrapper not initialized")

    try:
        runtime_class_map = dict(_wrapper.class_alias_map)
        runtime_class_map.update({k.lower(): v.lower() for k, v in req.class_alias_map.items()})

        runtime_rel_map = dict(_wrapper.relation_alias_map)
        runtime_rel_map.update({k.lower(): v.lower() for k, v in req.relation_alias_map.items()})

        local_wrapper = ContextAwareContentWrapper(
            exp_dir=_get_env("G2S_EXP_DIR"),
            epoch=_get_env("G2S_EPOCH", "100"),
            classes_txt_path=_get_env("G2S_CLASSES_PATH"),
            relationships_txt_path=_get_env("G2S_RELATIONSHIPS_PATH"),
            class_alias_map=runtime_class_map,
            relation_alias_map=runtime_rel_map,
            device=_get_env("G2S_DEVICE", "cuda"),
        )

        scene_nodes = local_wrapper.parse_scene_graph(req.scene_graph)
        additions = local_wrapper.parse_architect_additions(req.architect_response)

        model_inputs = local_wrapper.build_model_inputs(
            scene_nodes=scene_nodes,
            scene_edges=req.scene_graph.get("edges", []),
            additions=additions,
            default_added_class=req.default_added_class,
        )

        return local_wrapper.predict_addition_boxes(model_inputs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
def reload_wrapper() -> Dict[str, str]:
    global _wrapper
    _wrapper = _build_wrapper()
    return {"status": "reloaded"}


if __name__ == "__main__":
    import uvicorn

    host = _get_env("G2S_HOST", "127.0.0.1")
    port = int(_get_env("G2S_PORT", "8000"))
    uvicorn.run("ContextAwareContentWrapper:app", host=host, port=port, reload=False)
