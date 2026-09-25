import os
import zipfile
from pathlib import Path

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    # Files and folders to include
    # We map local paths to the zip archive paths required by the competition
    
    team_name = "Rookies" 
    zip_name = f"{team_name}_submission.zip"
    zip_path = PROJECT_ROOT / zip_name
    
    print(f"Creating submission zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # 1. Output Files
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            file_path = PROJECT_ROOT / "output" / f
            if file_path.exists():
                print(f"Adding output/{f}...")
                zipf.write(file_path, arcname=f"output/{f}")
            else:
                print(f"WARNING: {file_path} not found. Did you run the pipeline?")
                
        # 2. Source Code
        business_dir = PROJECT_ROOT / "src" / "business_entity_resolution"
        if business_dir.exists():
            for root, _, files in os.walk(business_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = Path(root) / file
                        zip_arcname = f"code/business_entity_resolution/src/{file_path.relative_to(business_dir)}"
                        zipf.write(file_path, arcname=zip_arcname)
        
        # Also include scripts/
        scripts_dir = PROJECT_ROOT / "src" / "scripts"
        if scripts_dir.exists():
            for root, _, files in os.walk(scripts_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = Path(root) / file
                        zip_arcname = f"code/business_entity_resolution/scripts/{file_path.relative_to(scripts_dir)}"
                        zipf.write(file_path, arcname=zip_arcname)
                        
        # 3. Documentation and Requirements
        for f in ["README.md", "requirements.txt", "Documentation_template.md"]:
            file_path = PROJECT_ROOT / f
            if file_path.exists():
                print(f"Adding {f}...")
                # We put them under code/business_entity_resolution/
                zipf.write(file_path, arcname=f"code/business_entity_resolution/{f}")
            else:
                print(f"WARNING: {file_path} not found.")
                
    print(f"Success! Submission packaged to {zip_path}")

if __name__ == "__main__":
    main()
