import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

def download_blob_if_not_exists(blob_name: str, destination_path: Path):
    """
    Downloads a file from Azure Blob Storage if it doesn't already exist locally.

    Args:
        blob_name (str): The full path (including folders) to the blob in the container.
        destination_path (Path): The local file path where the blob should be saved.
    """
    if destination_path.exists():
        print(f"[INFO] File already exists, skipping download: {destination_path}")
        return

    # Ensure the destination directory exists
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        # Silently return if connection string is not set
        return

    container_name = "models"

    try:
        print(f"[INFO] Connecting to Azure Blob Storage to download {blob_name}...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(destination_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        print(f"[INFO] Successfully downloaded {blob_name} to {destination_path}")

    except Exception as e:
        print(f"[ERROR] Failed to download {blob_name}. Error: {e}")
        # If download fails, you might want to handle this, e.g., by stopping the app
        # or falling back to a default model if one exists.
        # We'll also check if a partial file was created and remove it.
        if destination_path.exists():
            os.remove(destination_path)

