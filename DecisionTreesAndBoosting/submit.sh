#!/bin/sh

set -e

SUBMIT_DIR="cs181-submit-dir"

rm -rf $SUBMIT_DIR
mkdir $SUBMIT_DIR
ls -1 | grep -v "^tfutils$" |\
        grep -v "^Makefile$" |\
        grep -v "^$SUBMIT_DIR$" |\
        grep -v "^.*\\.csv$" |\
        grep -v "^.*\\.dat$" |\
        grep -v "^.*\\.pyc$" |\
        grep -v "^.*~" | xargs -n 1 -IHERE cp -r HERE $SUBMIT_DIR

if [ -f /usr/local/bin/submit ]; then
    echo "Submitting the following files:"
    ls $SUBMIT_DIR
    /usr/local/bin/submit cs181 1 `pwd`/$SUBMIT_DIR
    echo "Done."
else
    echo "Must submit of FAS machines"
fi