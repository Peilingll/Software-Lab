#!/usr/bin/env python3
"""
Pipeline Runner - (v5 - Clean, Visualize, OBB-IFC)
- 05: Cleans noise, saves debug visualization
- 06: Reads JSON to create accurate OBB-based IFC
- "batch" mode: Auto-stops after grace period
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

class PipelineRunner:
    def __init__(self, base_dir=None):
        """Initialize paths and settings."""
        self.base_dir = Path(__file__).parent.resolve()
        
        # --- Script Paths (using 'script' singular) ---
        self.scripts_dir = self.base_dir / "script"
        self.sonata_script = self.base_dir / "sonata" / "sonata_full_pipeline.py"
        self.script_01 = self.scripts_dir / "01_filter_structure.py"
        self.script_02 = self.scripts_dir / "02_prep_for_ransac.py"
        self.script_03 = self.scripts_dir / "03_wall_ransac.py"
        self.script_04 = self.scripts_dir / "04_split_walls_corrected.py"
        self.script_05 = self.scripts_dir / "05_compute_bbox_corrected.py"
        self.script_06 = self.scripts_dir / "06_bbox_to_ifc.py"
        
        # --- Directory Paths ---
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.batches_dir = self.base_dir / "batches"
        
        # --- Root working directories (temporary) ---
        self.work_output = self.base_dir / "output"
        self.work_output_clean = self.base_dir / "output_clean"
        self.work_per_wall = self.base_dir / "per_wall"
        
        # Create required directories
        self.batches_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Session data for logging
        self.session_data = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "batches": [],
            "start_time": datetime.now().isoformat()
        }
        
        # Keep track of files processed in this session
        self.session_processed_files = set()
        print("Initialized new processing session.")

    def get_las_files(self, source="scans"):
        """Get all .las files from the source directory."""
        source_dir = self.data_dir / source
        if not source_dir.exists():
            print(f"Error: Directory {source_dir} does not exist")
            return []
        las_files = list(source_dir.glob("*.las"))
        return sorted(las_files)

    def get_batch_name(self, las_file):
        """Generate a unique batch name from the file stem."""
        return f"batch_{las_file.stem}"

    def run_command(self, cmd, cwd=None, timeout=1800):
        """Executes a subprocess command, streaming stdout/stderr."""
        try:
            working_dir = cwd or self.base_dir
            print(f"  [Exec] {' '.join(cmd)}")
            
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.base_dir) + os.pathsep + env.get('PYTHONPATH', '')
            
            # Start the process
            process = subprocess.Popen(
                cmd, cwd=working_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', env=env
            )
            
            # Stream the output with timeout
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    process.kill()
                    process.wait()
                    return False, f"Timeout after {timeout}s"
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"    > {output.strip()}")

            # Check final return code
            if process.returncode != 0:
                print(f"  [Error] Command failed, return code: {process.returncode}")
                return False, f"Return code {process.returncode}"

            return True, "Success"
        except Exception as e:
            print(f"  [Error] Unexpected error: {e}")
            return False, str(e)

    def pre_run_cleanup(self, batch_name):
        """Clean up temporary directories and files from the root."""
        print(f"  Cleaning root working directories...")
        shutil.rmtree(self.work_output, ignore_errors=True)
        shutil.rmtree(self.work_output_clean, ignore_errors=True)
        shutil.rmtree(self.work_per_wall, ignore_errors=True)
        
        # Clean up any leftover files from a previous failed run
        for f in self.base_dir.glob(f"building_{batch_name}.ifc"):
            f.unlink(missing_ok=True)
        for f in self.base_dir.glob(f"wall_bboxes_{batch_name}.*"):
            f.unlink(missing_ok=True)

    def archive_batch_results(self, batch_name, batch_storage_dir, visualization_dir=None):
        """Move all temporary results into the batch-specific folder."""
        print(f"  Archiving results to {batch_storage_dir}")
        batch_storage_dir.mkdir(exist_ok=True)
        
        try:
            # Move the temporary directories
            if self.work_output.exists():
                shutil.move(str(self.work_output), str(batch_storage_dir / "output"))
            if self.work_output_clean.exists():
                shutil.move(str(self.work_output_clean), str(batch_storage_dir / "output_clean"))
            if self.work_per_wall.exists():
                shutil.move(str(self.work_per_wall), str(batch_storage_dir / "per_wall"))

            # Move the visualization directory
            if visualization_dir and visualization_dir.exists():
                # We rename it to be cleaner inside the batch folder
                shutil.move(str(visualization_dir), str(batch_storage_dir / "bbox_visuals"))

            # Move the final batch files
            ifc_file = self.base_dir / f"building_{batch_name}.ifc"
            if ifc_file.exists():
                shutil.move(str(ifc_file), str(batch_storage_dir / ifc_file.name))
                
            json_file = self.base_dir / f"wall_bboxes_{batch_name}.json"
            if json_file.exists():
                shutil.move(str(json_file), str(batch_storage_dir / json_file.name))

            csv_file = self.base_dir / f"wall_bboxes_summary_{batch_name}.csv"
            if csv_file.exists():
                shutil.move(str(csv_file), str(batch_storage_dir / csv_file.name))

        except Exception as e:
            print(f"  [Error] Failed to archive {batch_name}: {e}")

    def process_single_batch(self, las_file, batch_name):
        """Run the full 01-06 pipeline for a single .las file."""
        print(f"\n{'='*60}")
        print(f"Processing Batch: {batch_name} (Source: {las_file.name})")
        print(f"{'='*60}")
        
        batch_storage_dir = self.batches_dir / batch_name
        
        # Skip if this batch folder already exists
        if batch_storage_dir.exists():
            print(f"  Batch {batch_name} already exists. Skipping.")
            self.session_processed_files.add(las_file.name)
            if not any(b['batch_name'] == batch_name for b in self.session_data["batches"]):
                self.session_data["batches"].append({
                    "batch_name": batch_name, "source_file": str(las_file),
                    "status": "skipped (exists)", "output_dir": str(batch_storage_dir)
                })
                self.save_session_data()
            return True

        # --- Define paths for this batch ---
        viz_dir = self.base_dir / f"temp_viz_{batch_name}" # Temporary name
        viz_archive_name = "bbox_visuals" # Final name in archive

        try:
            # 1. Cleanup
            self.pre_run_cleanup(batch_name)
            shutil.rmtree(viz_dir, ignore_errors=True) # Clean temp viz dir
            
            # 2. Sonata
            print("\n[1/7] Running Sonata Semantic Segmentation...")
            success, _ = self.run_command([sys.executable, str(self.sonata_script), str(las_file),
                                          "--output-dir", str(self.work_output)])
            if not success: raise Exception("Sonata (1/7) failed")
            
            sonata_output_npz = self.work_output / "merged_classification.npz"
            if not sonata_output_npz.exists():
                 raise Exception(f"Sonata output not found: {sonata_output_npz}")

            # 3. 01_filter_structure
            print("\n[2/7] Running 01_filter_structure...")
            success, _ = self.run_command([
                sys.executable, str(self.script_01), 
                str(sonata_output_npz), "--out", str(self.work_output_clean)
            ])
            if not success: raise Exception("01_filter_structure (2/7) failed")
            wall_ply = self.work_output_clean / "walls.ply"
            if not wall_ply.exists(): raise Exception(f"walls.ply not found")
            floor_ply = self.work_output_clean / "floor_ceiling.ply"

            # 4. 02_prep_for_ransac
            print("\n[3/7] Running 02_prep_for_ransac...")
            success, _ = self.run_command([sys.executable, str(self.script_02), str(wall_ply)])
            if not success: raise Exception("02_prep_for_ransac (3/7) failed")
            wall_prep_ply = self.work_output_clean / "walls_prep.ply"
            if not wall_prep_ply.exists(): raise Exception(f"walls_prep.ply not found")

            # 5. 03_wall_ransac
            print("\n[4/7] Running 03_wall_ransac...")
            success, _ = self.run_command([sys.executable, str(self.script_03), str(wall_prep_ply)])
            if not success: raise Exception("03_wall_ransac (4/7) failed")
            ransac_ply = self.work_output_clean / "walls_ransac_colored.ply"
            if not ransac_ply.exists(): raise Exception(f"walls_ransac_colored.ply not found")

            # 6. 04_split_walls_corrected
            print("\n[5/7] Running 04_split_walls_corrected...")
            plane_csv = self.work_output_clean / "walls_ransac_planes.csv"
            cmd_04 = [
                sys.executable, str(self.script_04),
                str(ransac_ply), "--output", str(self.work_per_wall),
                "--planes-csv", str(plane_csv)
            ]
            success, _ = self.run_command(cmd_04)
            if not success: raise Exception("04_split_walls_corrected (5/7) failed")
            if not self.work_per_wall.exists(): raise Exception(f"per_wall directory not found")

             # 7. 05_compute_bbox_corrected (Cleaning & Visualizing)
            print("\n[6/7] Running 05_compute_bbox_corrected (Cleaning & Visualizing)...")
            batch_json_file = self.base_dir / f"wall_bboxes_{batch_name}.json" 
            
            cmd_05 = [
                sys.executable, str(self.script_05),
                "--input", str(self.work_per_wall),
                "--output", str(self.base_dir),
                "--suffix", batch_name,
                "--visualize-dir", str(viz_dir) 
            ]
            success, _ = self.run_command(cmd_05)
            if not success: raise Exception("05_compute_bbox_corrected (6/7) failed")
            
            
            if not batch_json_file.exists():
                raise Exception(f"BBox JSON not found: {batch_json_file}")

            # 8. 06_bbox_to_ifc_from_json (讀取 JSON)
            print("\n[7/7] Running 06_bbox_to_ifc (from JSON + Manhattan)...")
            batch_ifc_file = self.base_dir / f"building_{batch_name}.ifc"
            
        
            cmd_06 = [
                sys.executable, str(self.script_06),
                "--input-json", str(batch_json_file),   
                "--output", str(batch_ifc_file)
            ]
            
        
            if floor_ply.exists():
                cmd_06.extend(["--floor-ply", str(floor_ply)])
            
            success, _ = self.run_command(cmd_06)
            if not success: raise Exception("06_bbox_to_ifc (7/7) failed")
            
            # 9. Archive results
            self.archive_batch_results(batch_name, batch_storage_dir, viz_dir)

            # 10. Log success
            final_ifc_path = str(batch_storage_dir / batch_ifc_file.name)
            batch_info = {
                "batch_name": batch_name, "source_file": str(las_file),
                "timestamp": datetime.now().isoformat(), "status": "completed",
                "output_dir": str(batch_storage_dir),
                "final_ifc": final_ifc_path,
                "visuals": str(batch_storage_dir / viz_archive_name) # Log path
            }
            self.session_data["batches"].append(batch_info)
            self.save_session_data()
            self.session_processed_files.add(las_file.name)
            
            print(f"✓ Batch {batch_name} processed successfully")
            print(f"✓ Final IFC saved to: {final_ifc_path}")
            print(f"✓ BBox visuals saved to: {batch_info['visuals']}")
            return True
            
        except Exception as e:
            print(f"✗ FAILED to process batch {batch_name}: {e}")
            self.archive_batch_results(batch_name, batch_storage_dir, viz_dir) 
            batch_info = {"batch_name": batch_name, "status": "failed", "error": str(e)}
            self.session_data["batches"].append(batch_info)
            self.save_session_data()
            return False
        finally:
            self.pre_run_cleanup(batch_name)
            shutil.rmtree(viz_dir, ignore_errors=True)


    def batch_mode(self, source="scans", grace_period=15):
        """
        Auto-stopping batch mode.
        """
        print(f"\n--- Starting Auto-Batch Mode (Grace Period: {grace_period}s) ---")
        
        try:
            while True:
                print(f"\nScanning {self.data_dir / source}...")
                
                las_files = self.get_las_files(source)
                new_files_to_process = [f for f in las_files if f.name not in self.session_processed_files]

                if not new_files_to_process:
                    print("No new files found.")
                else:
                    print(f"Found {len(new_files_to_process)} new files to process:")
                    for f in new_files_to_process: print(f"  - {f.name}")
                    
                    for i, las_file in enumerate(new_files_to_process, 1):
                        print(f"\n--- Processing {i}/{len(new_files_to_process)} ---")
                        batch_name = self.get_batch_name(las_file)
                        self.process_single_batch(las_file, batch_name)

                print(f"\nProcessing complete. Waiting {grace_period}s for new files...")
                time.sleep(grace_period)
                
                las_files_after_wait = self.get_las_files(source)
                new_files_after_wait = [f for f in las_files_after_wait if f.name not in self.session_processed_files]
                
                if not new_files_after_wait:
                    print("No new files found during grace period. Exiting.")
                    break
                else:
                    print(f"✓ Found {len(new_files_after_wait)} new files during wait. Continuing...")
        
        except KeyboardInterrupt:
            print("\nUser interrupted (Ctrl+C). Exiting.")

        print(f"\n{'='*60}")
        print(f"Batch processing finished!")
        print(f"All results are in: {self.batches_dir}/")

    def watch_mode(self, source="scans", interval=30):
        """
        Permanent watch mode.
        """
        print(f"\n{'='*60}")
        print(f"--- Starting Permanent Watch Mode ---")
        print(f"--- (Interval: {interval}s) ---")
        print("Press Ctrl+C to stop.")
        print(f"{'='*60}\n")
        
        try:
            while True:
                las_files = self.get_las_files(source)
                new_files = [f for f in las_files if f.name not in self.session_processed_files]
                
                if new_files:
                    print(f"\n✓ Found {len(new_files)} new files, starting...")
                    for las_file in new_files:
                        batch_name = self.get_batch_name(las_file)
                        self.process_single_batch(las_file, batch_name)
                    print("\nFinished processing queue, returning to watch...")
                else:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] No new files. Waiting... (Total processed: {len(self.session_processed_files)})")
                
                print(f"⏰ Sleeping for {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"User stopped (Ctrl+C). Watch mode terminated.")
            print(f"{'='*60}")

    def save_session_data(self):
        """Save session data to a log file."""
        session_file = self.logs_dir / f"{self.session_data['session_id']}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [Warning] Could not save session log: {e}")

def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(description="Point Cloud to IFC Pipeline (v5 - OBB)")
    parser.add_argument("--source", default="scans", help="Source directory (relative to data/)")
    parser.add_argument("--mode", choices=["batch", "watch"], default="batch", 
                       help="batch (default): Auto-stop | watch: Permanent")
    parser.add_argument("--interval", type=int, default=15,
                       help="Grace period (batch) or poll interval (watch) in seconds.")
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    scripts_to_check = [runner.sonata_script, runner.script_01, runner.script_05, runner.script_06]
    for script_path in scripts_to_check:
        if not script_path.exists():
            print(f"Error: Missing critical script: {script_path}")
            print("Please ensure all scripts (01-06, sonata) are in the correct 'script/' and 'sonata/' directories.")
            sys.exit(1)
        
    if args.mode == "watch":
        runner.watch_mode(args.source, args.interval)
    else:
        runner.batch_mode(args.source, grace_period=args.interval)

if __name__ == "__main__":
    main()