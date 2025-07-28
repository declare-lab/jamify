from huggingface_hub import snapshot_download
import subprocess

if __name__ == "__main__":
    paths = snapshot_download(repo_id="declare-lab/jam-0.5")
    subprocess.run([
        "python", "-m", "jam.infer",
        f"evaluation.checkpoint_path={paths}/jam-0_5.safetensors"
    ])