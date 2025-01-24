import requests
import zipfile
import os
from tqdm import tqdm

def get_data_model():
    print('==== Downloading pre-trained models & Nordland images ====')
    # Download the pre-trained models
    dropbox_urls = [
        "https://www.dropbox.com/scl/fi/vb8ljhp5rm1cx4tbjfxx3/VPRTempo_pretrained_models.zip?rlkey=felsqy3qbapeeztgkfcdd1zix&st=xncoy7rg&dl=0",
        "https://www.dropbox.com/scl/fi/445psdi7srbhuqa807lyn/spring.zip?rlkey=5ciswaz0ygv107e6pzvxnm6ga&st=tdaslyuc&dl=0",
        "https://www.dropbox.com/scl/fi/l2dmccham4ifj0xf9p9jw/fall.zip?rlkey=gvmt5jvzdfw8p7008yfoxeb4s&st=z14ngqyx&dl=0",
        "https://www.dropbox.com/scl/fi/8ff3ozh6kujbg1vbnrasw/summer.zip?rlkey=563t03cd2vwfr32i9llg945m8&st=t96te8py&dl=0"
    ]

    folders = [
        "./vprtempo/models/",
        "./vprtempo/dataset/",
        "./vprtempo/dataset/",
        "./vprtempo/dataset/"
    ]

    names = [
        "VPRTempo_pretrained_models.zip",
        "spring.zip",
        "fall.zip",
        "summer.zip"
    ]

    for idx, url in enumerate(dropbox_urls):
        download_extract(url, folders[idx], names[idx])

    print('==== Downloading pre-trained models & Nordland images completed ====')

def download_extract(url, folder, name):
    # Ensure the destination folder exists
    os.makedirs(folder, exist_ok=True)

    # Modify the URL for direct download
    direct_download_url = url.replace("dl=0", "dl=1")

    # Send a HEAD request to get the total file size
    with requests.head(direct_download_url, allow_redirects=True) as head:
        if head.status_code != 200:
            print(f"Failed to retrieve header for {name}. Status code: {head.status_code}")
            return
        total_size = int(head.headers.get('content-length', 0))

    # Initialize the progress bar for downloading
    with requests.get(direct_download_url, stream=True) as response, \
         open(os.path.join(folder, name), "wb") as file, \
         tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {name}", ncols=80) as progress_bar:
        
        if response.status_code != 200:
            print(f"Failed to download {name}. Status code: {response.status_code}")
            return

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    # Determine extraction path
    if name == "VPRTempo_pretrained_models.zip":
        extract_path = folder
    else:
        extract_path = os.path.join(folder, name.replace('.zip', ''))

    # Open the zip file
    with zipfile.ZipFile(os.path.join(folder, name), 'r') as zip_ref:
        # Get list of files in the zip
        members = zip_ref.namelist()
        # Initialize the progress bar for extraction
        with tqdm(total=len(members), desc=f"Extracting {name}", unit='file', ncols=80) as extract_bar:
            for member in members:
                zip_ref.extract(member, path=extract_path)
                extract_bar.update(1)

    # Remove the zip file after extraction
    os.remove(os.path.join(folder, name))
    print(f"Completed {name}")