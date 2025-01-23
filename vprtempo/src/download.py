import requests
import zipfile
import os

def get_data_model():
    print('==== Downloading pre-trained models & Nordland images ====')
    # Download the pre-trained models
    dropbox_urls = [
    "https://www.dropbox.com/scl/fi/ko8ghx3qigv7ao59yitaa/VPRTempo_pretrained_models.zip?rlkey=i3yhgwmh3a7hqi2bkilm4li18&st=upqgcanu&dl=0",
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
    # check if folder exists, if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    direct_download_url = url.replace("dl=0", "dl=1")

    # Download the file using requests
    response = requests.get(direct_download_url, stream=True)

    # Save the file locally
    with open(f'{folder}{name}', "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    # unzip the downloaded file
    if name == "VPRTempo_pretrained_models.zip":
        with zipfile.ZipFile(f'{folder}{name}', 'r') as zip_ref:
            zip_ref.extractall(f'{folder}')
    else:
        with zipfile.ZipFile(f'{folder}{name}', 'r') as zip_ref:
            zip_ref.extractall(f'{folder}{name}'.replace('.zip', ''))

    # delete the zip file
    os.remove(f'{folder}{name}')