import os
import requests
import zipfile
import json

# check if the data folder exists
os.makedirs("data", exist_ok=True)

datasets_info = json.load(open("datasets_info.json", "r"))

for dataset in datasets_info:
    if "path" in dataset:
        if os.path.isfile(dataset["path"]):
            print(f"Dataset [{dataset['name']}] already exists.\n")
            continue
        else:
            print(f"Dataset [{dataset['name']}] not found, downloading...\n")
    else:
        if dataset["download_url"].endswith(".zip"):
            print(f"Downloading [{dataset['name']}].")
            response = requests.get(dataset["download_url"])
            zip_path = os.path.join("data", f"{dataset['name']}.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("data")
            os.remove(zip_path)
            if dataset["name"] == "text8":
                dataset["path"] = os.path.join("data", dataset["name"])
            elif dataset["name"] == "The Analects":
                dataset["path"] = os.path.join("data", "lunyu.txt")
            dataset["size"] = os.path.getsize(dataset["path"])
            print(f"Completed downloading.\n")
        else:
            print(f"Downloading [{dataset['name']}].")
            response = requests.get(dataset["download_url"])
            dataset_path = os.path.join("data", f"{dataset['name'].replace(' ', '_')}.txt")
            with open(dataset_path, "wb") as f:
                f.write(response.content)
            dataset["path"] = dataset_path
            dataset["size"] = os.path.getsize(dataset_path)
            print(f"Completed downloading.\n")

json.dump(datasets_info, open("datasets_info.json", "w"), indent=4)