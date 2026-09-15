import json
import logging
import os
from dataclasses import asdict

logger = logging.getLogger(__name__)

def dataclass_write_json_file(obj, file_path:str, file_name:str, indent:int=2):
    """
    Use this function to neatly persist a dataclass config object as a JSON file.
    """
    obj_dict = asdict(obj)

    try: 
        os.makedirs(file_path)
    except FileExistsError:
        pass
    
    file_path_name = os.path.join(file_path, file_name)

    with open(file_path_name, 'w') as json_file:
        json.dump(obj_dict, json_file, indent=indent)
        logger.info(f"Wrote JSON config file to: {file_path_name}")