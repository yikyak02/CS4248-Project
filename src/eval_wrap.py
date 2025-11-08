# src/eval_wrap.py
import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", type=str, required=True)
    ap.add_argument("--pred_json", type=str, required=True)
    ap.add_argument("--evaluator_path", type=str, required=True,
                    help="Path to official evaluate-v2.0.py (or v1.1 eval) script")
    args = ap.parse_args()

    cmd = [sys.executable, args.evaluator_path, args.dev_json, args.pred_json]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(completed.stdout)

if __name__ == "__main__":
    main()
