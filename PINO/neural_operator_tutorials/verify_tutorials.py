"""Validate structure and execute each notebook in its own Jupyter kernel."""
import argparse
import asyncio
import ast
import json
import os
import sys
# Avoid two Intel OpenMP runtimes in Windows Anaconda kernels.
# This is inherited by subprocess kernels before NumPy is imported.
if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from pathlib import Path
import time
import traceback
import nbformat
from nbclient import NotebookClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke","full"], default="smoke")
    parser.add_argument("--save", action="store_true", help="Write actual executed outputs back")
    parser.add_argument("--start", type=int, default=0, help="First lesson number, inclusive")
    parser.add_argument("--end", type=int, default=99, help="Last lesson number, inclusive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.environ["MPLBACKEND"] = "Agg"
    if args.mode == "smoke":
        os.environ["NO_TUTORIAL_SMOKE"] = "1"
        os.environ["NO_TUTORIAL_EPOCHS"] = "2"
    else:
        os.environ.pop("NO_TUTORIAL_SMOKE",None)
        os.environ.pop("NO_TUTORIAL_EPOCHS",None)
    records = []
    paths = sorted(root.glob("[0-9][0-9]_*.ipynb"))
    if not 0 <= args.start <= args.end <= 99:
        parser.error("Require 0 <= start <= end <= 99")
    paths = [p for p in paths if args.start <= int(p.name[:2]) <= args.end]
    if not paths:
        raise RuntimeError("No notebooks selected")
    selection = "" if (args.start,args.end)==(0,99) else f"_{args.start:02d}_{args.end:02d}"
    for path in paths:
        start = time.perf_counter()
        record = {"notebook":path.name,"mode":args.mode}
        print(f"Running {path.name} ({args.mode})",flush=True)
        try:
            nb = nbformat.read(path,as_version=4)
            nbformat.validate(nb)
            for index,cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    ast.parse(cell.source,filename=f"{path.name}:cell{index}")
            client = NotebookClient(nb,timeout=600,kernel_name="python3",
                                    resources={"metadata":{"path":str(root)}})
            executed = client.execute()
            record["status"] = "passed"
            record["stdout"] = "".join(
                output.get("text","")
                for cell in executed.cells if cell.cell_type=="code"
                for output in cell.get("outputs",[]) if output.output_type=="stream")
            record["figures"] = sum(
                "image/png" in output.get("data",{})
                for cell in executed.cells if cell.cell_type=="code"
                for output in cell.get("outputs",[]))
            if record["figures"] == 0:
                raise RuntimeError("Expected inline figures, found none")
            if args.save:
                nbformat.write(executed,path)
            print(record["stdout"][-1500:],flush=True)
        except Exception:
            record["status"] = "failed"
            record["error"] = traceback.format_exc()
            print(record["error"],flush=True)
        record["seconds"] = round(time.perf_counter()-start,2)
        records.append(record)
        (root/f"verification{selection}_{args.mode}.json").write_text(
            json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    failures = [r for r in records if r["status"] != "passed"]
    print(f"{len(records)-len(failures)}/{len(records)} passed",flush=True)
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
