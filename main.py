
import subprocess
import sys

STAGES = [
    ["-m", "src.data_loader"],
    ["-m", "src.eda"],
    ["-m", "src.train"],
    ["-m", "src.tune"],
    ["-m", "src.evaluate"]
]

for stage in STAGES:
    print(f"\nRunning: {' '.join(stage)}")

    subprocess.run(
        [sys.executable, *stage],
        check=True
    )
