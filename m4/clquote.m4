# clquote.m4 -- add double quotation marks on cl file   -*- Autoconf -*-
# This file contains a custom macro

AC_DEFUN([CL_QUOTE_VERSION], [1.0])dnl

# shell script for adding double quotation marks
AC_DEFUN([CL_QUOTE_SH],
[$SED -e '/^builddir = .*/i\cl_quote_sh_script = $(top_builddir)\/$1\/$2' \ 
      -e '/^builddir = .*/i\cl_quote_sh = $(SHELL) $(cl_quote_sh_script)' \
./Makefile > ./Makefile.tmp
$MV ./Makefile.tmp ./Makefile
])dnl

# directory for cl file without double quotation marks
AC_DEFUN([CL_NOQUOTE_DIR],
[$SED '/^builddir = .*/i\cl_noquote_dir = $(top_builddir)\/$1' \
./Makefile > ./Makefile.tmp
$MV ./Makefile.tmp ./Makefile
])dnl

# directory for cl file with double quotation marks
AC_DEFUN([CL_QUOTE_DIR],
[$SED '/^builddir = .*/i\cl_quote_dir = $(top_builddir)\/$1' \
./Makefile > ./Makefile.tmp
$MV ./Makefile.tmp ./Makefile
])dnl

# clean cl file with double quotation marks
AC_DEFUN([CLEAN_CL_QUOTE_FILE],
[$SED -e '/^distclean: .*/i\clean-cl-quote-file: \
\t -rm -rf $(cl_quote_dir)\n' \
      -e 's/^clean: \(.*\)$/\nclean: clean-cl-quote-file \1/' \
      -e 's/^distclean: \(.*\)$/distclean: clean-cl-quote-file \1/' \
./Makefile > ./Makefile.tmp
$MV ./Makefile.tmp ./Makefile
])dnl

AC_DEFUN([CL_QUOTE],
[$SED 's/^install: \(.*\)$/ \
install:\
\t @if test ! -d $(cl_quote_dir); then \\\
\t   $(MKDIR_P) $(cl_quote_dir); fi; \\\
\t abs_cl_list=`find $(cl_noquote_dir) -name "*.cl"`; \\\
\t for abs_cl_name in $${abs_cl_list}; do \\\
\t   cl_name=`basename $${abs_cl_name}`; \\\
\t   if test -f $(cl_quote_dir)\/$${cl_name}; then \\\
\t     if test $${abs_cl_name} -nt \\\
\t       $(cl_quote_dir)\/$${cl_name}; then \\\
\t         $(cl_quote_sh) \\\
\t         $(cl_noquote_dir)\/$${cl_name} \\\
\t         $(cl_quote_dir)\/$${cl_name}; \\\
\t     fi; \\\
\t   else \\\
\t     $(cl_quote_sh) \\\
\t     $(cl_noquote_dir)\/$${cl_name} \\\
\t     $(cl_quote_dir)\/$${cl_name}; \\\
\t   fi; \\\
\t done\
\t @$(MAKE) \1/' ./Makefile > ./Makefile.tmp
$MV ./Makefile.tmp ./Makefile
])dnl
