from pathlib import Path

# Simulate what happens in the webapp
current_file = Path("src/webapp/pages/realtime_mapping.py").resolve()
project_root = current_file.parent.parent.parent

print("Current file:", current_file)
print("Project root:", project_root)
print("Models path:", project_root / "models" / "weights")
print("GT path:", project_root / "outputs" / "ground_truth")
print("Models exists:", (project_root / "models" / "weights").exists())
print("GT exists:", (project_root / "outputs" / "ground_truth").exists())

# Also check what we actually have
actual_root = Path(".")
print("\nActual paths:")
print("Actual models:", (actual_root / "models" / "weights").exists())
print("Actual GT:", (actual_root / "outputs" / "ground_truth").exists())
