import subprocess
from pathlib import Path
import hailort  # Pour l'inférence Hailo après compilation

# --------------------------
# CONFIGURATION
# --------------------------
onnx_dir = Path("./onnx_models")      # Dossier contenant les ONNX
hef_dir = Path("./hef_models")        # Dossier de sortie pour HEF
calibration_data = Path("./sample_data")  # Dataset pour quantization (optionnel)
test_inference = True                 # True pour tester l'inférence après compilation

hef_dir.mkdir(parents=True, exist_ok=True)

# --------------------------
# FONCTION DE COMPILATION
# --------------------------
def compile_model(onnx_path: Path):
    model_name = onnx_path.stem
    ir_dir = hef_dir / f"{model_name}_ir"
    ir_dir.mkdir(parents=True, exist_ok=True)
    hef_path = hef_dir / f"{model_name}.hef"

    # 1️⃣ Conversion ONNX → IR avec quantization
    cmd_ir = [
        "dataflow_compiler",
        "--input-model", str(onnx_path),
        "--output-dir", str(ir_dir),
        "--target", "hailo8",
        "--precision", "int8"
    ]
    if calibration_data.exists():
        cmd_ir += ["--calibration-data", str(calibration_data)]

    print(f"[INFO] Converting {onnx_path} to IR...")
    try:
        subprocess.run(cmd_ir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] IR generation failed for {onnx_path}: {e}")
        return None

    # 2️⃣ Compiler IR → HEF
    cmd_hef = [
        "dataflow_compiler",
        "--compile-ir", str(ir_dir),
        "--output", str(hef_path)
    ]
    print(f"[INFO] Compiling IR to HEF...")
    try:
        subprocess.run(cmd_hef, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] HEF compilation failed for {onnx_path}: {e}")
        return None

    print(f"[SUCCESS] HEF generated at: {hef_path}")
    return hef_path

# --------------------------
# FONCTION TEST INFERENCE
# --------------------------
def test_inference_hef(hef_path: Path):
    print(f"[INFO] Testing inference for {hef_path}")
    network = hailort.Network(str(hef_path))
    network.load()
    # Exemple: entrée aléatoire compatible (adapter selon ton modèle)
    import numpy as np
    input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)
    output = network.infer(input_tensor)
    print(f"[INFO] Inference output keys: {list(output.keys())}")

# --------------------------
# MAIN
# --------------------------
def main():
    onnx_files = list(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"[ERROR] No ONNX files found in {onnx_dir}")
        return

    for onnx_file in onnx_files:
        hef_path = compile_model(onnx_file)
        if hef_path and test_inference:
            try:
                test_inference_hef(hef_path)
            except Exception as e:
                print(f"[ERROR] Inference failed for {hef_path}: {e}")

if __name__ == "__main__":
    main()