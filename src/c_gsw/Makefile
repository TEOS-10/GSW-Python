#
#  $Id: Makefile,v 1e5e75c749c2 2015/08/08 22:03:51 fdelahoyde $
#
               CFLAGS:=	-O3
            CINCLUDES:=
              Library:=	libgswteos-10.so
              Program:=	gsw_check
      $(Program)_SRCS:=	gsw_check_functions.c \
			gsw_oceanographic_toolbox.c \
			gsw_saar.c
      $(Library)_SRCS:=	gsw_oceanographic_toolbox.c \
			gsw_saar.c
      $(Library)_OBJS:=	gsw_oceanographic_toolbox.o \
			gsw_saar.o

all:	$(Program)

$(Program):	$($(Program)_SRCS)
	gcc $(CFLAGS) $(CINCLUDES) -o $(Program) $($(Program)_SRCS) -lm

library:	$(Library)

$(Library):	$($(Library)_SRCS)
	gcc -fPIC -c $(CFLAGS) $(CINCLUDES) $($(Library)_SRCS)
	gcc -shared -o $(Library) $($(Library)_OBJS) -lm

clean:
	rm -f $(Program) $(Library) $($(Library)_OBJS)
