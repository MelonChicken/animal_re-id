"""Execute the pilot's plain-Python notebook cells without installing Jupyter.

Run with the existing RoVF interpreter from the repository root::

    experiments/reference/rovf/.venv/Scripts/python.exe scripts/run_temporal_consistency_notebook.py

This small runner captures stdout, errors and matplotlib figures. It does not
implement magics, widgets, rich display or the Jupyter kernel protocol.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback


def main() -> None:
    # Zoo names contain characters outside the default Windows cp949 console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="notebooks/temporal_consistency_pilot.ipynb")
    parser.add_argument("--identity-count", type=int, help="Seeded subset of identities (default: all)")
    parser.add_argument("--all-test-clips", action="store_true", help="Use every Test clip of the selected identities")
    args = parser.parse_args()
    if args.identity_count is not None and args.identity_count < 1:
        parser.error("--identity-count must be positive")
    root = Path(__file__).resolve().parents[1]
    source = root / args.notebook
    notebook = json.loads(source.read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overrides = {}
    if args.identity_count is not None:
        overrides["identity_count"] = args.identity_count
    if args.all_test_clips:
        overrides["clips_per_identity"] = None
    namespace = {"__name__": "__main__", "PILOT_ROOT": root, "PILOT_OVERRIDES": overrides}
    target = None
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        outputs = cell["outputs"] = []
        class Capture(io.StringIO):
            def write(self, value):
                sys.__stdout__.write(value)
                sys.__stdout__.flush()
                return super().write(value)

        stream = Capture()

        def show(*_args, **_kwargs):
            for number in plt.get_fignums():
                buffer = io.BytesIO()
                plt.figure(number).savefig(buffer, format="png", bbox_inches="tight", dpi=110)
                outputs.append({"output_type": "display_data", "metadata": {}, "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": ["<matplotlib figure>"]}})
            plt.close("all")

        plt.show = show
        failed = False
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                exec(compile("".join(cell["source"]), f"{source.name}:cell-{count}", "exec"), namespace)
            except Exception as error:
                failed = True
                outputs.append({"output_type": "error", "ename": type(error).__name__,
                                "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
        if stream.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(True)})
        if "RUN_DIR" in namespace:
            target = namespace["RUN_DIR"] / source.name
        if target is not None:
            target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        if failed:
            raise RuntimeError(f"Cell {count} failed: {outputs[-1].get('evalue')}; executed notebook: {target}")
    print(f"Executed notebook: {target}")


if __name__ == "__main__":
    main()
