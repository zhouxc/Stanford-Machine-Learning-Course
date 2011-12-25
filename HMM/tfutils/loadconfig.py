#!/usr/bin/env python

"""
loadconfig.py -- manages assignment configuration options
"""

try:
    import json
except ImportError:
    import simplejson as json
import os
from os import path

REQUIRED_FIELDS = ("title", "testmodule", "workmodule", "taskmodule")
DEFAULT_FIELDS = {"subtitle":"", "assignment_number": None,
                  "course_name": None}

DEFAULT_CONFIG_NAME = "config.js"

def load_config_file(sDir,sConfigFile=DEFAULT_CONFIG_NAME):
    if sConfigFile not in set(os.listdir(sDir)):
        raise ValueError("%s not found in %s, could not load config file."
                         % (sConfigFile, sDir))
    dictConfig = dict(DEFAULT_FIELDS)
    try:
        infile = open(path.join(sDir,sConfigFile))
        dictConfig.update(json.load(infile))
    finally:
        infile.close()
    for sRequired in REQUIRED_FIELDS:
        if sRequired not in dictConfig:
            raise ValueError("Required field '%s' not found in %s."
                             % (sRequired, sConfigFile))
    dictConfig["config_filename"] = sConfigFile
    return dictConfig

def get_config_dir(sConfigFile=DEFAULT_CONFIG_NAME):
    sPath = path.dirname(path.abspath(__file__))
    while sPath.strip('/') and sConfigFile not in os.listdir(sPath):
        sPath = path.dirname(sPath)
    if not sPath.strip('/'):
        return None
    return sPath

__all__ = [load_config_file]

if __name__ == "__main__":
    pass
