# UMI Visualizer

A high-performance visualizer for Universal Manipulation Interface (UMI) datasets using Rerun.

## Installation

This project uses `uv` for dependency management.

1. **Install uv** (for detailed installation guide refer [Install uv](URL "[https://docs.astral.sh/uv/getting-started/installation/]")):
   ```bash
   curl -sSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

2. **Install System Dependencies (Linux only):**
   ```bash
   sudo apt-get update && sudo apt-get install -y libblosc-dev liblz4-dev

3. **Run the Visualizer** (Make sure to include the dataset.zarr in your path):
   ```bash
   uv run main.py /path/to/your/dataset.zarr --episode 1

4. **Use [SPACEBAR] to start or pause the Visualization**

5. **Close the Rerun Window and press Ctrl+C in the terminal to stop the execution**

<img width="1849" height="1057" alt="image" src="https://github.com/user-attachments/assets/5127c2ed-447b-4647-81c8-4439de6eb290" />

   
