from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io_utils import ensure_out_dir, write_json, write_text
from ..release import inspect_release, render_release_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect CausalDiagSafe release files without training or loading image pixels."
    )
    parser.add_argument("--config", required=True, help="Release-path JSON configuration.")
    parser.add_argument("--out-dir", required=True, help="Small report output directory.")
    parser.add_argument(
        "--skip-image-existence",
        action="store_true",
        help="Skip sequential image path existence checks.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("release config root must be an object")

    summary = inspect_release(config, check_images=not args.skip_image_existence)
    out_dir = ensure_out_dir(args.out_dir)
    write_json(out_dir / "release_inspection.json", summary)
    write_text(out_dir / "release_inspection.md", render_release_markdown(summary))
    print(
        f"release inspection complete: blocking={summary['blocking_issue_count']} "
        f"warnings={summary['warning_count']} out={out_dir}",
        flush=True,
    )
    if summary["blocking_issue_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
