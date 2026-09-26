"""
Package the final submission zip for the Amazon ML Challenge.

Expected zip structure:
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/business_entity_resolution/
  │   ├── src/           (all Python source files)
  │   ├── scripts/       (pipeline scripts)
  │   ├── README.md
  │   └── requirements.txt
  └── Documentation_template.md
"""
import os
import zipfile
from pathlib import Path


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    team_name = "Rookies"
    zip_path = PROJECT_ROOT / f"{team_name}_submission.zip"

    print(f"Creating submission: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── 1. Output files ──
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = PROJECT_ROOT / "output" / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"output/{fname}")
                size_mb = fpath.stat().st_size / 1024 / 1024
                print(f"  [+] output/{fname}  ({size_mb:.1f} MB)")
            else:
                print(f"  [!] output/{fname}  NOT FOUND — run the pipeline first!")

        # ── 2. Source code (src/) ──
        src_dir = PROJECT_ROOT / "src" / "business_entity_resolution"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.endswith(".py"):
                        fp = Path(root) / f
                        arc = f"code/business_entity_resolution/src/{fp.relative_to(src_dir)}"
                        zf.write(fp, arcname=arc)
                        print(f"  [+] {arc}")

        # ── 3. Scripts ──
        scripts_dir = PROJECT_ROOT / "src" / "scripts"
        if scripts_dir.exists():
            for root, _, files in os.walk(scripts_dir):
                for f in files:
                    if f.endswith(".py"):
                        fp = Path(root) / f
                        arc = f"code/business_entity_resolution/scripts/{fp.relative_to(scripts_dir)}"
                        zf.write(fp, arcname=arc)
                        print(f"  [+] {arc}")

        # ── 4. README + requirements ──
        for fname in ["README.md", "requirements.txt"]:
            fpath = PROJECT_ROOT / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"code/business_entity_resolution/{fname}")
                print(f"  [+] code/business_entity_resolution/{fname}")

        # ── 5. Documentation template (goes at zip root) ──
        doc = PROJECT_ROOT / "Documentation_template.md"
        if doc.exists():
            zf.write(doc, arcname="Documentation_template.md")
            print(f"  [+] Documentation_template.md")

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\n[+] Submission packaged: {zip_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
