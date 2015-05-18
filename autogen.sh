#!/bin/sh

ln -s ../../tools/pre-commit-code-style.sh .git/hooks/pre-commit

echo "git submodule update"
git submodule sync
git submodule init
git submodule update

echo copying m4/clquote.m4 to /usr/share/aclocal
cp m4/clquote.m4 /usr/share/aclocal

echo "Generating configure files"
autoreconf -i
# Run twice to get around a "ltmain.sh" bug
autoreconf --install --force
