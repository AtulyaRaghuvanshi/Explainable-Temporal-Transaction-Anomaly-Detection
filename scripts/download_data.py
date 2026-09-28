from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
root=Path(__file__).resolve().parents[1];out=root/'data'/'raw';out.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate();api.competition_download_files('ieee-fraud-detection',path=str(out));z=out/'ieee-fraud-detection.zip'
if z.exists():
    with zipfile.ZipFile(z) as f:f.extractall(out)
