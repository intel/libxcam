#! /bin/sh

CL_DIR=../cl_kernel/

echo "---- add double quotation in cl file ----"
for filename in `ls ${CL_DIR}*.cl`
do
  awk '
     BEGIN {FS="[" " \t+]"}
     {
        if($1~/\// || !NF || $0~/^[ ]+$/ || $1=="" && $2~/\*/ || $2=="" && $3~/\*/)
           print $0;
        else 
        {
           gsub(/\\n\"$/, "");
           gsub(/\"$/, "");
           gsub(/^\"/, "");

           print "\""$0"\\n\"";
         }
     }
     ' $CL_DIR$filename > $CL_DIR$filename.tmp

  mv $CL_DIR$filename.tmp $CL_DIR$filename 
done
echo "---- add double quotation in cl file done ----"

