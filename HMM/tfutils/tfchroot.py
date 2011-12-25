#!/usr/bin/env python

"""tfchroot.py -- create a chroot environment suitable for testing
student-submitted code."""

import glob
import inspect
import os
from os import path
import shutil
import subprocess
import sys
import tarfile
import unittest

import pygments
import pygments.lexers
import pygments.formatters

import loadconfig

REQUIRED_MODULES = ["unittest", "math"]

def retrieve_files(mod):
    listSFile = []
    listSFile.append(path.abspath(mod.__file__))
    for o in inspect.getmembers(mod):
        if inspect.ismodule(o):
            listSFile.extend(retrieve_files(o))
    return listSFile

def enumerate_dependencies():
    sExecutable = path.abspath(sys.executable)
    listSFile = [sExecutable]
    for sMod in REQUIRED_MODULES:
        mod = __import__(sMod)
        listSFile.extend(retrieve_files(mod))
    return listSFile

def copy_to_chroot(sChrootDir, sFile):
    sChrootFile = sChrootDir.rstrip('/') + sFile
    sChrootFileDir = path.dirname(sChrootFile)
    if not path.exists(sChrootFileDir):
        os.makedirs(sChrootFileDir)
    shutil.copy(sFile,sChrootFile)

def build_chroot(sDirBase, sConfigDir, dictConfig):
    sDir = path.join(sConfigDir, sDirBase)
    if not path.exists(sDir):
        os.makedirs(sDir)
    for sFile in enumerate_dependencies():
        copy_to_chroot(sDir,sFile)
    for sKey in ("testmodule",):
        sFileBase = dictConfig[sKey]
        sFile = path.join(sConfigDir,sFileBase)
        print sFile,sDir
        shutil.copy(sFile,path.join(sDir,sFileBase))
    return sDir

def extract_base_name(sArchiveFile):
    return path.splitext(path.basename(sArchiveFile))[0]

def extract_work_module(sArchiveFile, sWorkMod):
    sBase = extract_base_name(sArchiveFile)
    try:
        tf = tarfile.open(sArchiveFile)
        infile = tf.extractfile(path.join("%s.submit" % sBase, sWorkMod))
        sRet = infile.read()
    finally:
        infile.close()
        tf.close()
    return sRet

def list_archives(sArchiveDir, sConfigDir):
    sFullArchiveDir = path.join(sConfigDir, sArchiveDir)
    return glob.glob(path.join(sFullArchiveDir, "*.tgz"))

def color_module(sModuleContents):
    return pygments.highlight(sModuleContents, pygments.lexers.PythonLexer(),
                              pygments.formatters.HtmlFormatter())

def dump_to_html_file(sHtml, sTitle, sOutputFile):
    sStyle = pygments.formatters.HtmlFormatter().get_style_defs(".highlight")
    sOut = """<!DOCTYPE html>
    <html>
    <head><title>%(sTitle)s</title><style>
    div.highlight {background-color:#FFFFFF;}
    %(sStyle)s
    </style></head>
    <body><h1>%(sTitle)s</h1>%(sHtml)s</body></html>
    """ % {"sTitle":sTitle, "sStyle": sStyle, "sHtml": sHtml}
    outfile = open(sOutputFile, "wb")
    try:
        outfile.write(sOut)
    finally:
        outfile.close()    

def dump_work_modules(sArchiveDir, sOutputDir, sConfigDir, sWorkMod):
    sFullOutputDir = path.join(sConfigDir, sOutputDir)
    if not path.exists(sFullOutputDir):
        os.makedirs(sFullOutputDir)
    for sArchiveFile in list_archives(sArchiveDir, sConfigDir):
        sBase = extract_base_name(sArchiveFile)
        sWorkModuleContents = extract_work_module(sArchiveFile, sWorkMod)
        sHtml = color_module(sWorkModuleContents)
        dump_to_html_file(sHtml, "Work module for %s" % sBase,
                          path.join(sFullOutputDir, "%s.html" % sBase))

def prepare_test(sDirBase, sArchiveDir):
    sConfigDir = loadconfig.get_config_dir()
    dictConfig = loadconfig.load_config_file(sConfigDir)
    sChrootDir = build_chroot(sDirBase, sConfigDir, dictConfig)
    sWorkMod = dictConfig["workmodule"]
    for sFile in liat_archives(sArchiveDir, sConfigDir):
        sBase = path.splitext(path.basename(sFile))[0]
        sWorkModContents = extrac_work_module(sFile, sWorkMod)
        outfile = open(path.join(sChrootDir, sWorkMod), 'wb')
        outfile.write(sWorkModContents)
        break

def run_test(sDirBase, sModWork):
    sConfigDir = loadconfig.get_config_dir()
    sChrootDir = build_chroot(sDirBase, sConfigDir)
    shutil.copy(path.join(sConfigDir, path.basename(sModWork)), sDirBase)
    sDirOrig = os.getcwd()
    os.chroot(sChrootDir)
    unittest.main(__import__(sModWork))

def main(argv):
    if len(argv) > 1:
        sDir = argv[1]
    else:
        sDir = "testbin"
    #run_test(sDir, "testnn.py")
    #prepare_test(sDir, "grading")
    sConfigDir = loadconfig.get_config_dir()
    dump_work_modules("grading", "workmodules", sConfigDir, "dtree.py")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
