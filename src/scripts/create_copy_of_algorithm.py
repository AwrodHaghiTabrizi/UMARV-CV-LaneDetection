import os
import shutil
from datetime import datetime
import logging
import sys
import json
import nbformat
import re

repo_dir = os.getcwd()

src_dir = os.path.join(repo_dir, "src")
sys.path.insert(0, src_dir)
from helpers import *

def main():

    creation_date = datetime.now()
    algorithm_id_b10 = int(creation_date.strftime("%y%m%d%H%M%S"))
    algorithm_id = base10_to_base36(algorithm_id_b10)

    copy_algorithm_id = input("Insert algorithm id to copy: ")
    copy_algorithm_dir = f"{repo_dir}/algorithms/algorithm_{copy_algorithm_id}"
    if not os.path.exists(copy_algorithm_dir):
        print(f"Algorithm {copy_algorithm_id} not found")
        return

    new_algorithm_dir = f"{repo_dir}/algorithms/algorithm_{algorithm_id}"
    copy_directory(
        source_dir = copy_algorithm_dir,
        destination_dir = new_algorithm_dir
    )

    algorithm_info_dir = f"{new_algorithm_dir}/content/info.json"
    with open(algorithm_info_dir, 'r') as file:
        algorithm_info = json.load(file)
    algorithm_info['author'] = input("Insert algorithm author: ")
    algorithm_info['algorithm_id'] = algorithm_id
    algorithm_info['creation_date'] = creation_date.strftime("%B %d, %Y, %H:%M:%S")
    with open(algorithm_info_dir, 'w') as file:
        json.dump(algorithm_info, file, indent=4)

    notebook_names = ["colab_env", "lambdalabs_env", "mac_env", "windows_env", "jetson_env"]
    for notebook_name in notebook_names:
        notebook_dir = f"{new_algorithm_dir}/src/notebooks/{notebook_name}.ipynb"
        with open(notebook_dir, 'r') as file:
            notebook = nbformat.read(file, as_version=4)
        cell = notebook['cells'][3]
        pattern = r"os\.environ\[\"ALGORITHM_ID\"\] = \"[a-zA-Z0-9]+\""
        match = re.search(pattern, cell["source"])
        if match:
            old_algorithm_id = match.group()
            cell["source"] = cell["source"].replace(old_algorithm_id, f"os.environ[\"ALGORITHM_ID\"] = \"{algorithm_id}\"")
        with open(notebook_dir, 'w') as file:
            nbformat.write(notebook, file)

    print(f"Algorithm {algorithm_id} initialized")

if __name__ == "__main__":
    main()