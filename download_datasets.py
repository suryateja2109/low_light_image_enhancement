import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_lol():
    raw_dir = r"data\raw"
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "lol_dataset.zip")
    url = "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading LOL dataset from {url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="lol_dataset.zip") as pbar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
        print("Download complete.")
    else:
        print(f"{zip_path} already exists.")
        
    print("Extracting lol_dataset.zip...")
    extract_dir = r"data\raw\lol_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    download_and_extract_lol()
