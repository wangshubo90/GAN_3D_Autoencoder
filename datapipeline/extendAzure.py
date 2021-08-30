# azureml-core of version 1.0.72 or higher is required
import os
from azureml.core import Workspace, Dataset
from zipfile import ZipFile

def download_dataset(workspace, datasetName, root_dir=None):
    """
    workspace: Azure ML workspace 
    """
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), "data")
    dataset = Dataset.get_by_name(workspace, name='extend')
    dataset.download(target_path=root_dir, overwrite=False)

def unpack(file, dstdir=None):
    with ZipFile('file', 'r') as f:
        f.extractall("dstdir")

if __name__=="__main__":
    workspace = ""
    datasetName = ""

    download_dataset(workspace, datasetName)
