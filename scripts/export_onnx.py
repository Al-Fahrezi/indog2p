# scripts/export_onnx.py
import os
import sys
import yaml
import json
import torch
from transformers import BertConfig, BertForMaskedLM

CONFIG_PATH = "config/config.yaml"
VOCAB_PATH = "config/vocab.json"
CHECKPOINT_PATH = "checkpoints/bert_g2p_best.pt"
ONNX_OUTPUT = "checkpoints/bert_g2p.onnx"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        d = json.load(f)
    return d["token2id"], d["id2token"], d["vocab"]

def main():
    print("=== Export BERT G2P to ONNX ===")
    config = load_config()
    token2id, id2token, vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = BertConfig(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["num_heads"],
        num_hidden_layers=config["model"]["num_layers"],
        intermediate_size=config["model"]["feedforward_dim"],
        max_position_embeddings=config["model"]["max_len"],
        pad_token_id=token2id["[PAD]"],
    )
    model = BertForMaskedLM(model_cfg)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")
        sys.exit(1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    print("Model loaded.")

    # Dummy input: batch=1, seq_len=max_len (padding)
    max_len = config["model"]["max_len"]
    dummy_input = torch.randint(
        low=0, high=len(vocab),
        size=(1, max_len),
        dtype=torch.long
    ).to(device)

    # Ekspor ONNX
    print(f"Exporting to {ONNX_OUTPUT} ...")
    torch.onnx.export(
        model,
        (dummy_input,),
        ONNX_OUTPUT,
        export_params=True,
        opset_version=17,  # opset terbaru, sesuaikan dengan environment
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    print("ONNX export selesai.")

    # Validasi ONNX (opsional)
    try:
        import onnx
        import onnxruntime as ort
        print("Validasi hasil ONNX...")
        onnx_model = onnx.load(ONNX_OUTPUT)
        onnx.checker.check_model(onnx_model)
        print("Model ONNX valid.")

        # Uji prediksi PyTorch vs ONNX
        with torch.no_grad():
            pt_logits = model(dummy_input).logits.cpu().numpy()
        ort_session = ort.InferenceSession(ONNX_OUTPUT)
        ort_inputs = {'input_ids': dummy_input.cpu().numpy()}
        ort_logits = ort_session.run(['logits'], ort_inputs)[0]
        diff = abs(pt_logits - ort_logits).mean()
        print(f"Rata-rata selisih prediksi PyTorch vs ONNX: {diff:.6f}")
        assert diff < 1e-4, "Prediksi ONNX berbeda signifikan!"
        print("Prediksi ONNX dan PyTorch cocok.")
    except ImportError:
        print("ONNXRuntime/onnx belum terinstall. Lewati validasi.")
    except Exception as e:
        print(f"Error validasi ONNX: {e}")

    # Simpan tokenizer vocab (penting untuk deployment)
    vocab_out = ONNX_OUTPUT.replace(".onnx", "_vocab.json")
    with open(vocab_out, "w") as f:
        json.dump({"token2id": token2id, "id2token": id2token, "vocab": vocab}, f, indent=2)
    print(f"Vocab tokenizer disimpan ke: {vocab_out}")

if __name__ == "__main__":
    main()
