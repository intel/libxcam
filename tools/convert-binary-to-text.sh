#! /bin/sh
# Convert binary file to text file.
# Command line:
#    ./convert-binary-to-text.sh xxx.cl.bin xxx.cl.binx

BINARY_FILE=$1
BINARYX_FILE=$2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <binary_file> <text_file>\n"
    exit 1
fi

od -A n -t x1 -v $BINARY_FILE | \
    awk '
    BEGIN { print "{" }
    {
        printf "   "
        for (i = 1; i < NF; i++)
            { printf " 0x" $i "," }
        print " 0x" $i ","
    }
    END { print "};" }
    ' > $BINARYX_FILE.tmp

mv $BINARYX_FILE.tmp $BINARYX_FILE
