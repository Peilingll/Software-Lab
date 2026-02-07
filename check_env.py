#!/usr/bin/env python3
"""Environment verification script for Point Cloud to BIM Pipeline."""

import sys
import os

PASS = "PASS"
FAIL = "FAIL"

results = []


def check(name, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((name, status, detail))
    tag = f"[{status}]"
    msg = f"  {tag:6s} {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def main():
    print("=" * 60)
    print("Pipeline Environment Check")
    print("=" * 60)

    # 1. Python version
    v = sys.version_info
    check("Python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")

    # 2. Critical imports
    print("\n--- Python packages ---")
    packages = {
        "numpy": "numpy",
        "torch": "torch",
        "open3d": "open3d",
        "laspy": "laspy",
        "scipy": "scipy",
        "sklearn": "sklearn",
        "ifcopenshell": "ifcopenshell",
        "spconv": "spconv",
        "torch_scatter": "torch_scatter",
        "timm": "timm",
        "huggingface_hub": "huggingface_hub",
        "packaging": "packaging",
    }
    for display_name, mod_name in packages.items():
        try:
            __import__(mod_name)
            check(f"import {display_name}", True)
        except ImportError as e:
            check(f"import {display_name}", False, str(e))

    # 3. CUDA
    print("\n--- CUDA ---")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            check("CUDA available", True, f"{gpu_name}, {mem_gb:.1f} GB")
        else:
            check("CUDA available", False, "CPU only")
    except Exception as e:
        check("CUDA available", False, str(e))

    # 4. Checkpoints
    print("\n--- Model checkpoints ---")
    ckpts = [
        "ckpt/sonata.pth",
        "ckpt/sonata_linear_prob_head_sc.pth",
    ]
    for ckpt in ckpts:
        exists = os.path.isfile(ckpt)
        check(f"Checkpoint {ckpt}", exists)

    # 5. Directory structure
    print("\n--- Directory structure ---")
    dirs = ["sonata/", "script/", "data/"]
    for d in dirs:
        exists = os.path.isdir(d)
        check(f"Directory {d}", exists)

    # 6. Pipeline scripts
    print("\n--- Pipeline scripts ---")
    scripts = [
        "sonata/sonata_full_pipeline.py",
        "script/01_filter_structure.py",
        "script/02_prep_for_ransac.py",
        "script/03_wall_ransac.py",
        "script/04_split_walls_corrected.py",
        "script/05_compute_bbox_corrected.py",
        "script/06_bbox_to_ifc.py",
    ]
    for s in scripts:
        exists = os.path.isfile(s)
        check(f"Script {s}", exists)

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = total - passed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("\nFailed checks:")
        for name, status, detail in results:
            if status == FAIL:
                msg = f"  - {name}"
                if detail:
                    msg += f": {detail}"
                print(msg)
    else:
        print("All checks passed!")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
