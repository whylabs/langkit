# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none
import logging
import os
import zipfile
from dataclasses import dataclass
from typing import cast

import requests
import whylabs_client
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from whylabs_client.api.assets_api import AssetsApi
from whylabs_client.model.get_asset_response import GetAssetResponse

from langkit.config import LANGKIT_CACHE

logger = logging.getLogger(__name__)

configuration = whylabs_client.Configuration(host="https://api.whylabsapp.com")
configuration.api_key["ApiKeyAuth"] = os.environ["WHYLABS_API_KEY"]
configuration.discard_unknown_keys = True

client = whylabs_client.ApiClient(configuration)
assets_api = AssetsApi(client)


@dataclass
class AssetPath:
    asset_id: str
    tag: str
    zip_path: str
    extract_path: str


def _get_asset_path(asset_id: str, tag: str = "0") -> AssetPath:
    return AssetPath(
        asset_id=asset_id,
        tag=tag,
        zip_path=f"{LANGKIT_CACHE}/assets/{asset_id}/{tag}/{asset_id}.zip",
        extract_path=f"{LANGKIT_CACHE}/assets/{asset_id}/{tag}/{asset_id}",
    )


def _is_extracted(asset_id: str, tag: str = "0") -> bool:
    asset_path = _get_asset_path(asset_id, tag)
    if not os.path.exists(asset_path.zip_path):
        return False

    with zipfile.ZipFile(asset_path.zip_path, "r") as zip_ref:
        zip_names = set(zip_ref.namelist())
        extract_names = set(os.listdir(asset_path.extract_path))
    return zip_names.issubset(extract_names)


def _extract_asset(asset_id: str, tag: str = "0"):
    asset_path = _get_asset_path(asset_id, tag)
    with zipfile.ZipFile(asset_path.zip_path, "r") as zip_ref:
        zip_ref.extractall(asset_path.extract_path)


def _is_zip_file(file_path: str) -> bool:
    try:
        with zipfile.ZipFile(file_path, "r"):
            return True
    except zipfile.BadZipFile:
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(max=5))
def _download_asset(asset_id: str, tag: str = "0"):
    asset_path = _get_asset_path(asset_id, tag)
    response: GetAssetResponse = cast(GetAssetResponse, assets_api.get_asset(asset_id))
    url = cast(str, response.download_url)
    os.makedirs(os.path.dirname(asset_path.zip_path), exist_ok=True)
    r = requests.get(url, stream=True)
    with open(asset_path.zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)

    if not _is_zip_file(asset_path.zip_path):
        os.remove(asset_path.zip_path)
        raise ValueError(f"Downloaded file {asset_path.zip_path} is not a zip file")


def get_asset(asset_id: str, tag: str = "0"):
    asset_path = _get_asset_path(asset_id, tag)
    if _is_extracted(asset_id, tag):
        logger.info(f"Asset {asset_id} with tag {tag} already downloaded and extracted")
        return asset_path.extract_path

    if not os.path.exists(asset_path.zip_path):
        logger.info(f"Downloading asset {asset_id} with tag {tag} to {asset_path.zip_path}")
        _download_asset(asset_id, tag)

    logger.info(f"Extracting asset {asset_id} with tag {tag}")
    _extract_asset(asset_id, tag)
    return asset_path.extract_path
