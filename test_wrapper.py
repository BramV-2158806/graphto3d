import json
import os
import sys

# Ensure current directory is in python path to handle local imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ContextAwareContentWrapper import ContextAwareContentWrapper


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Environment Setup
    # --------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set default environment variables if not already set.
    # These defaults assume the directory structure described in README.md
    if not os.getenv("G2S_EXP_DIR"):
        # Default to a shared experiment folder; adjust if your model is elsewhere
        os.environ["G2S_EXP_DIR"] = os.path.join(base_dir, "experiments", "shared_model")
    
    if not os.getenv("G2S_CLASSES_PATH"):
        os.environ["G2S_CLASSES_PATH"] = os.path.join(base_dir, "GT", "classes.txt")

    if not os.getenv("G2S_RELATIONSHIPS_PATH"):
        os.environ["G2S_RELATIONSHIPS_PATH"] = os.path.join(base_dir, "GT", "relationships.txt")

    if not os.getenv("G2S_DEVICE"):
        # Fallback to CPU for testing if CUDA is not explicitly requested/available
        os.environ["G2S_DEVICE"] = "cpu"

    print(f"Using Experiment Dir: {os.environ['G2S_EXP_DIR']}")
    print(f"Using Device: {os.environ['G2S_DEVICE']}")

    # --------------------------------------------------------------------------
    # 2. Initialize Wrapper
    # --------------------------------------------------------------------------
    try:
        wrapper = ContextAwareContentWrapper(
            exp_dir=os.environ["G2S_EXP_DIR"],
            epoch=os.getenv("G2S_EPOCH", "100"),
            classes_txt_path=os.environ["G2S_CLASSES_PATH"],
            relationships_txt_path=os.environ["G2S_RELATIONSHIPS_PATH"],
            device=os.environ["G2S_DEVICE"],
            add_scene_node=True
        )
    except Exception as e:
        print(f"\n[Error] Failed to initialize wrapper: {e}")
        print("Tip: Ensure you have downloaded the 3DSSG data (GT folder) and model checkpoints (experiments folder).")
        return

    # --------------------------------------------------------------------------
    # 3. Load Input Data
    # --------------------------------------------------------------------------
    scene_graph_path = os.path.join(base_dir, "scene_graph.json")
    architect_path = os.path.join(base_dir, "architect_response_cache.json")

    print(f"Loading scene graph: {scene_graph_path}")
    with open(scene_graph_path, "r") as f:
        scene_graph = json.load(f)

    print(f"Loading architect response: {architect_path}")
    with open(architect_path, "r") as f:
        architect_response = json.load(f)

    # --------------------------------------------------------------------------
    # 4. Run Prediction Pipeline
    # --------------------------------------------------------------------------
    print("Running prediction...")
    try:
        scene_nodes = wrapper.parse_scene_graph(scene_graph)
        additions = wrapper.parse_architect_additions(architect_response)
        
        print(f"parsed {len(scene_nodes)} scene nodes and {len(additions)} additions.")

        model_inputs = wrapper.build_model_inputs(
            scene_nodes=scene_nodes,
            scene_edges=scene_graph.get("edges", []),
            additions=additions,
            default_added_class="picture" # Fallback class
        )

        response = wrapper.predict_addition_boxes(model_inputs)

        print("\nPrediction Result:")
        print(response.json(indent=2))
    except Exception as e:
        print(f"\n[Error] Prediction failed: {e}")

if __name__ == "__main__":
    main()