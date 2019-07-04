
OPENCVINC = `pkg-config --cflags opencv4`
OPENCVLNK = `pkg-config --libs opencv4`

CCDEBUG = -g -Ofast -Wall -Wno-write-strings
CCINCS = -I/usr/include -I. -I/usr/local/include $(OPENCVINC)
CCLIBS = -L/usr/lib/x86_64-linux-gnu -L. -L/usr/local/lib $(OPENCVLNK)
CCLINK = -ljpeg -lmpfit -lcfitsio -lm -lpthread

GCC=gcc
CC=g++


OFILES=mpfit.o
LIBFILE=libmpfit.a

all: $(LIBFILE) idscheck 	

mpfit.o: mpfit.c mpfit.h
	$(GCC) -c -o $@ $< $(CFLAGS)

$(LIBFILE): $(OFILES)
	$(AR) -r $@ $(OFILES)
	
clean:
	rm -f *.o 

idscheck.o: idscheck.cpp
	$(CC) $(CCDEBUG) $(CCINCS) -o $@ -c $< $(CFLAG)

idscheck: idscheck.o
	$(CC) -o $@ $< $(CFLAG) $(CCLIBS) $(CCLINK)

	
pushprocyon:
	rsync -avuz /Users/chyan/Documents/workspace/IDSControl/* -e ssh chyan@procyon:/home/chyan/IDSControl/
	
pushspica:
	rsync -avuz ../IDSControl -e ssh chyan@spica:/home/chyan/
