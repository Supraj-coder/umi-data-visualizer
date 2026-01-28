# UMI Visualizer

A high-performance visualizer for Universal Manipulation Interface (UMI) datasets using Rerun.

## Installation

1. **Generate your "dataset.zarr" data from the UMI repository**
   (Reference [UMI repository](URL "https://github.com/real-stanford/universal_manipulation_interface"))

This project uses `uv` for dependency management.

2. **Install uv** (for detailed installation guide refer [Install uv](URL "[https://docs.astral.sh/uv/getting-started/installation/]")):
   ```bash
   curl -sSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

3. **Install System Dependencies (Linux only):**
   ```bash
   sudo apt-get update && sudo apt-get install -y libblosc-dev liblz4-dev

4. **Run the Visualizer**:
   ```bash
   uv run main.py /path/to/your/dataset.zarr --episode 1

5. **Use [SPACEBAR] to start or pause the Visualization**

6. **Close the Rerun Window and press Ctrl+C in the terminal to stop the execution**

7. **Running tests:**
   ```bash
   uv run black .
   uv run ruff check . --fix
   uv run mypy main.py
   uv run -m pytest
****
<img width="1849" height="1057" alt="image" src="https://github.com/user-attachments/assets/5127c2ed-447b-4647-81c8-4439de6eb290" />

   




