#!/usr/bin/env python3
"""Fetch and patch the 5G-LENA revision required by the research scenario."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
NR_DIR = ROOT / "contrib" / "nr"
NR_REPOSITORY = "https://gitlab.com/cttc-lena/nr.git"
NR_COMMIT = "9b830f9adfc28bb559f4eaa5fc22fd4518e1cdfa"


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    if NR_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {NR_DIR}")

    NR_DIR.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", NR_REPOSITORY, str(NR_DIR))
    run("git", "checkout", NR_COMMIT, cwd=NR_DIR)

    patch_groups = [
        "official_energy_v5_2026-10-19/000*.patch",
        "true_bwp_dci_v6_2026-10-19/0001-*.patch",
        "true_bwp_dci_v6_2026-10-19/0002-*.patch",
        "slot_start_eligibility_2026-10-23/0001-*.patch",
        "slot_start_eligibility_2026-10-23/0002-*.patch",
        "slot_start_eligibility_2026-10-23/0004-*.patch",
    ]
    for pattern in patch_groups:
        patches = sorted((ROOT / "experiment_archive").glob(pattern))
        if not patches:
            raise SystemExit(f"No patch matched experiment_archive/{pattern}")
        for patch in patches:
            run(
                "git",
                "-c",
                "user.name=CI 5G-LENA Setup",
                "-c",
                "user.email=ci-5g-lena@example.invalid",
                "am",
                str(patch),
                cwd=NR_DIR,
            )

    runtime_patches = sorted(
        (ROOT / "experiment_archive/true_bwp_dci_v6_2026-10-19").glob(
            "0003-nr-runtime-api-completion.patch"
        )
    )
    if len(runtime_patches) != 1:
        raise SystemExit("Expected exactly one runtime API completion patch")
    run("git", "apply", str(runtime_patches[0]), cwd=NR_DIR)


if __name__ == "__main__":
    main()
