import os

__default_data_folder = "langkit_data"


def get_data_home() -> str:
    global __default_data_folder
    data_folder = os.getenv("LANGKIT_DATA_FOLDER", __default_data_folder)
    package_dir: str = os.path.dirname(__file__)
    data_path: str = os.path.join(package_dir, data_folder)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return data_path


__ALL__ = [get_data_home]
