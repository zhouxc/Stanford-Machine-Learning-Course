#!/usr/bin/env python

"""
updatemanager.py -- checks for updates to tfutils on GitHub and retrieves them.
"""

import datetime
import json
import os
from os import path
import re
import tarfile
import time
import urllib

import modules

VERSIONS_DIR_PATH = path.join(path.dirname(path.abspath(__file__)), "versions")
TARBALL_DIR_PATH = path.join(VERSIONS_DIR_PATH, "tarballs")
BACKUP_DIR_PATH = path.join(VERSIONS_DIR_PATH, "backups")
ORIGIN_CONFIG_PATH = path.join(VERSIONS_DIR_PATH, "origin.js")
VERSIONS_INFO_PATH = path.join(VERSIONS_DIR_PATH, "versioninfo.js")
GITHUB_DOMAIN = "github.com"
UNPACK_DIR_PATH = path.dirname(path.dirname(path.abspath(__file__)))
BACKUP_SOURCE_DIR_PATH = path.dirname(path.abspath(__file__))

TARBALL_RE = re.compile(r"^tarball_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}"
                        r"\.tar\.gz$")
TIMEZONE_RE = re.compile(r"[+]|[-]\d{2}[:]?\d{2}$")

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
def parse_dt(sDt):
    sDtClean,sTz = TIMEZONE_RE.split(sDt, 1)
    # TODO(jhoon): incorporate the timezone information.
    return datetime.datetime.strptime(sDtClean, DATETIME_FORMAT)

class Commit(object):
    def __init__(self, id, sMessage, dt):
        self.id = id
        self.sMessage = sMessage
        self.dt = dt
    def to_json(self):
        return {"id":self.id, "message": self.sMessage,
                "committed_date": self.dt.isoformat() + "-0000"}
    @classmethod
    def from_json(self, dictCommit):
        dt = parse_dt(dictCommit["committed_date"])
        return Commit(dictCommit["id"], dictCommit["message"], dt)
    @classmethod
    def empty(cls):
        dt = datetime.datetime.fromtimestamp(0.0)
        return Commit("", "(No commit found.)", dt)
    def __repr__(self):
        return "Commit(%r,%r,%r)" % (self.id, self.sMessage, self.dt)

def github_api_call(*listArgs):
    listSPieces = [GITHUB_DOMAIN, "api", "v2", "json"] + list(listArgs)
    return "http://" + path.join(*listSPieces)

def github_last_commit(sUser,sRepo,sBranch):
    sUrl = github_api_call("commits", "list", sUser, sRepo, sBranch)
    infile = None
    try:
        infile = urllib.urlopen(sUrl)
        dictJson = json.load(infile)
    finally:
        infile.close()
    listCommit = map(Commit.from_json, dictJson["commits"])
    listCommit.sort(lambda a,b: -cmp(a.dt,b.dt))
    return (listCommit and listCommit[0]) or None

def tarball_filename():
    dt = datetime.datetime.now()
    tpl = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    return "tarball_%04d_%02d_%02d_%02d_%02d_%02d.tar.gz" % tpl

def github_tarball(sUser,sRepo,sBranch):
    return "https://" + path.join(GITHUB_DOMAIN, sUser, sRepo, "tarball",
                                  sBranch)

def load_origin_config(sPath=ORIGIN_CONFIG_PATH):
    with open(sPath) as infile:
        return json.load(infile)

def load_version_commit(sPath=VERSIONS_INFO_PATH):
    if not path.isfile(sPath):
        return Commit.empty()
    with open(sPath) as infile:
        return Commit.from_json(json.load(infile))

def latest_commit(dictOriginConfig, sBranchType):
    sUser = dictOriginConfig["user"]
    sRepo = dictOriginConfig["repo"]
    sBranch = dictOriginConfig["branches"][sBranchType]
    return github_last_commit(sUser,sRepo,sBranch)

def is_update_available(cmtLatest,cmtVersion):
    return cmtLatest if cmtLatest.dt > cmtVersion.dt else None

def download_tarball(dictOriginConfig,sBranchType):
    sUser = dictOriginConfig["user"]
    sRepo = dictOriginConfig["repo"]
    sBranch = dictOriginConfig["branches"][sBranchType]
    sUrl = github_tarball(sUser,sRepo,sBranch)
    sFilename = tarball_filename()
    if not path.exists(TARBALL_DIR_PATH):
        os.makedirs(TARBALL_DIR_PATH)
    sFullFilename = path.join(TARBALL_DIR_PATH, sFilename)
    urllib.urlretrieve(sUrl, sFullFilename)
    return sFullFilename

def clean_downloads():
    for sFilename in os.listdir(TARBALL_DIR_PATH):
        if TARBALL_RE.match(sFilename) is not None:
            sFullPath = path.join(TARBALL_DIR_PATH, sFilename)
            os.unlink(sFullPath)

def splitall(s):
    listComp = []
    while s:
        sPref,sBase = path.split(s)
        listComp.append(sBase)
        s = sPref
    listComp.reverse()
    return listComp

def rootname(s):
    listAll = splitall(s)
    if len(listAll) >= 2:
        return listAll[1]
    return ""

def is_relevant_file(s):
    return rootname(s) in modules.TFUTILS_FILES
             
def check_for_updates(dictOriginConfig=None,sBranchType="master"):
    if dictOriginConfig is None:
        dictOriginConfig = load_origin_config()
    cmtLatest = latest_commit(dictOriginConfig, sBranchType)
    cmtVersion = load_version_commit()
    return is_update_available(cmtLatest,cmtVersion)

def unpack_tarball(sTarballFilename, sUnpackOnto, sUser, sRepo):
    sReSrc = r'^%s-%s-\w+' % (re.escape(sUser), re.escape(sRepo))
    rePrefix = re.compile(sReSrc)
    sPref = "tfutils"
    tf = tarfile.open(sTarballFilename)
    for ti in tf.getmembers():
        ti.name = rePrefix.sub(sPref,ti.name,1)
        if is_relevant_file(ti.name):
            tf.extract(ti,sUnpackOnto)
        
def backup_name():
    dt = datetime.datetime.now()
    return ("backup_%04d_%02d_%02d_%02d_%02d_%02d.tar.gz" %
            (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))

def filter_backup(sPath):
    return (BACKUP_DIR_PATH in sPath
            or TARBALL_DIR_PATH in sPath
            or "/build" in sPath
            or ".git/" in sPath
            or sPath.endswith(".pyc"))

def backup_current(sRoot):
    if not path.exists(BACKUP_DIR_PATH):
        os.makedirs(BACKUP_DIR_PATH)
    sFullFilename = path.join(BACKUP_DIR_PATH, backup_name())
    tf = tarfile.open(sFullFilename,"w:gz")
    tf.add(sRoot, path.basename(sRoot), exclude=filter_backup)
    tf.close()
    return sFullFilename

def update_version_info(cmt,sPath=VERSIONS_INFO_PATH):
    dictJs = cmt.to_json()
    with open(sPath, "wb") as outfile:
        json.dump(dictJs,outfile)

def has_git(sPath=BACKUP_SOURCE_DIR_PATH):
    return path.exists(path.join(sPath, ".git"))      

def deploy_updates():
    if has_git():
        return False
    dictOriginConfig = load_origin_config()
    sBranchType = "master"
    cmt = check_for_updates(dictOriginConfig, sBranchType)
    if cmt is None:
        return False
    backup_current(BACKUP_SOURCE_DIR_PATH)
    sFilename = download_tarball(dictOriginConfig, sBranchType)
    unpack_tarball(sFilename, UNPACK_DIR_PATH, dictOriginConfig["user"],
                   dictOriginConfig["repo"])
    update_version_info(cmt)
    return True

def main(argv):
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-c", "--check", action="store_true", dest="check",
                      help="check for updates")
    parser.add_option("-b", "--backup", action="store_true", dest="backup",
                      help="create a backup tarball")
    parser.add_option("-r", "--remove-downloaded", action="store_true",
                      dest="remove_downloaded")
    parser.add_option("-d", "--deploy", action="store_true",
                      dest="deploy")
    parser.add_option("-g", "--git", action="store_true", dest="git",
                      help="determine if this is a git working copy.")
    opts,args = parser.parse_args(argv)
    if opts.git:
        if has_git():
            print "This is a git working copy."
        else:
            print "Not a git working copy."
    if opts.check:
        cmt = check_for_updates()
        if cmt is None:
            print "No updates available."
        else:
            sDt = cmt.dt.strftime("%B %d, %Y at %I:%M %p")
            print "Update available, released %s" % sDt
    if opts.backup:
        sFilename = backup_current(BACKUP_SOURCE_DIR_PATH)
        print "Backup created at %s" % sFilename
    if opts.remove_downloaded:
        clean_downloads()
    if opts.deploy:
        if not deploy_updates():
            print "No updates found."
            return 1
        print "Update successful."
    return 0
        

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
