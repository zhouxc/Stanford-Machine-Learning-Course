#!/usr/bin/env python

"""
buildassignment.py -- packages an assignment into a tarball for distribution.
"""

import glob
import logging
import optparse
import os
from os import path
import shutil
import StringIO
import subprocess
import sys
import tarfile
import unittest

import loadconfig
import modules
import publish
import tftask

MODULE_KEYS = ("testmodule", "workmodule", "taskmodule",)

class MissingFileException(Exception):
    def __init__(self, sDir, sFile, sMsg):
        super(MissingFileException,self).__init__(sMsg)
        self.sDir = sDir
        self.sFile = sFile

def fileimport(sModuleName):
    return __import__(path.splitext(sModuleName)[0])

def check_directory(sDir, dictConfig):
    setContents = set(os.listdir(sDir))
    for sKey in MODULE_KEYS:
        sFilename = dictConfig[sKey]
        if sFilename not in setContents:
            raise MissingFileException(sDir, sFilename,"required code module")
    for sFilename in dictConfig.get("static",()):
        if sFilename not in setContents:
            raise MissingFileException(sDir, sFilename, "static file")
        
def build_tar(sDir, dictConfig, outfile):
    tfile = tarfile.open(mode="w:gz", fileobj=outfile)
    tfile.close()

def create_build_dir(sSrcDir, dictConfig, sDestDir):
    if sDestDir is None:
        sDestDir = path.abspath(sSrcDir)
    sDestDir = path.abspath(path.join(path.dirname(path.abspath(__file__)),
                                      "build",
                                      path.basename(sDestDir)))
    if path.isfile(sDestDir):
        raise ValueError("File exists where build directory should be at %s"
                         % sDestDir)
    elif not path.exists(sDestDir):
        os.makedirs(sDestDir)
    else:
        shutil.rmtree(sDestDir)
        os.makedirs(sDestDir)
    return sDestDir

def build_call_graph(sSrcDir,sDestDir,dictConfig):
    logging.info("Building call graph")
    sOutputFile = path.join(sDestDir,"call_graph.png")
    sTestFile = path.join(sSrcDir, dictConfig["testmodule"])
    sWorkName = path.splitext(dictConfig["workmodule"])[0]
    subprocess.check_call(("pycallgraph",
                           "--output-file=%s" % sOutputFile,
                           "--include=%s.*" % sWorkName,
                           sTestFile), cwd=sSrcDir)

def clean_work_module(sDestDir,dictConfig):
    def run_tests():
        reload(modtest)
        suite = unittest.TestLoader().loadTestsFromModule(modtest)
        buf = StringIO.StringIO()
        runner = unittest.TextTestRunner(verbosity=0,stream=buf)
        tr = runner.run(suite)
        buf.close()
        return len(tr.failures) + len(tr.errors), suite.countTestCases()

    logging.info("Cleaning work module")
    sWorkMod = path.splitext(dictConfig["workmodule"])[0]
    modtest = fileimport(dictConfig["testmodule"])
    logging.info("Running test suite (all tests should pass)")
    cBadTest,cTotal = run_tests()
    if cBadTest:
        raise ValueError("Tests failed.")
    modwork = fileimport(dictConfig["workmodule"])
    sCode = publish.clean_module(modwork)
    try:
        outfile = open(path.join(sDestDir,dictConfig["workmodule"]), "wb")
        outfile.write(sCode)
    finally:
        outfile.close()
    sys.path.insert(0,sDestDir)
    del sys.modules[sWorkMod]
    reload(modtest)
    logging.info("Running tests on cleaned work module (all test should fail)")
    cBadTest,cTotal = run_tests()
    if cBadTest != cTotal:
        raise ValueError("Tests passed after stripping work module.")
    del sys.modules[sWorkMod]
    sys.path.pop(0)

def copy_tfutils(sDestDir):
    sLocalSrcDir = path.abspath(path.dirname(__file__))
    sDestTfUtils = path.join(sDestDir, "tfutils")
    if not path.isdir(sDestTfUtils):
        os.makedirs(sDestTfUtils)
    for sTreeSrcSuff in modules.TFUTILS_FILES:
        sTreeSrc = path.join(sLocalSrcDir,sTreeSrcSuff)
        if path.isfile(sTreeSrc):
            sDest = path.join(sDestTfUtils,sTreeSrcSuff)
            sDestDir = path.dirname(sDest)
            if not path.isdir(sDestDir):
                os.makedirs(sDestDir)
            shutil.copy(sTreeSrc, sDest)
        else:
            shutil.copytree(path.join(sLocalSrcDir,sTreeSrcSuff),
                            path.join(sDestTfUtils,sTreeSrcSuff))

def build_tex_files(sSrcDir, sDestDir, dictConfig):
    for sFilename in dictConfig.get("texfiles",()):
        subprocess.check_call(("pdflatex",path.join(sSrcDir,sFilename)),
                              cwd=sSrcDir)
        sPdfFilename = path.splitext(sFilename)[0] + '.pdf'
        shutil.copy(path.join(sSrcDir, sPdfFilename),
                    path.join(sDestDir, sPdfFilename))
                          

def populate_build_dir(sSrcDir, sDestDir, dictConfig):
    logging.info("Populating build directory")
    build_tex_files(sSrcDir, sDestDir, dictConfig)
    clean_work_module(sDestDir,dictConfig)
    def move(sFilename):
        shutil.copy(path.join(sSrcDir,sFilename),
                    path.join(sDestDir,sFilename))
    for sKey in ("testmodule", "taskmodule"):
        move(dictConfig[sKey])
    for sFilename in dictConfig.get("static",()):
        move(sFilename)

    move(dictConfig["config_filename"])
    copy_tfutils(sDestDir)
    build_call_graph(sSrcDir, sDestDir, dictConfig)
    for sPycFilename in glob.glob(path.join(sDestDir, "*.pyc")):
        os.remove(sPycFilename)

def tar_build_dir(sDestDir):
    logging.info("Building tar archive")
    sOutfileName = sDestDir + ".clean.tar.gz"
    tfile = tarfile.open(sOutfileName, "w:gz")
    tfile.add(sDestDir, path.basename(sDestDir))
    tfile.close()
    
def main(argv):
    parser = optparse.OptionParser()
    parser.add_option("-o", "--outfile", action="store", type=str,
                      dest="outfile_name", default=None,
                      help="name of output file")
    opts,args = parser.parse_args(argv)
    if len(args) < 2:
        sDir = loadconfig.get_config_dir()
    else:
        sDir = args[1]
    sDir = path.abspath(sDir)
    sys.path.append(sDir)
    dictConfig = loadconfig.load_config_file(sDir)
    try:
        check_directory(sDir,dictConfig)
    except MissingFileException as mfexn:
        print ("Failed to find required file '%s' in directory %s"
               % (mfexn.sFile, mfexn.sDir))
        return 1
    sDestDir = create_build_dir(sDir,dictConfig,None)
    LOG_FILENAME = path.join(sDestDir,"build.log")
    logging.basicConfig(filename=LOG_FILENAME,level=logging.debug)
    populate_build_dir(sDir,sDestDir,dictConfig)
    tar_build_dir(sDestDir)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
