
OPENCVINC = `pkg-config --cflags opencv4`
OPENCVLNK = `pkg-config --libs opencv4`

CCDEBUG = -g -Ofast -Wall -Wno-write-strings
CCINCS = -I/usr/include -I. -I/usr/local/include $(OPENCVINC)
CCLIBS = -L/usr/lib/x86_64-linux-gnu -L. -L/usr/local/lib $(OPENCVLNK)
CCLINK = -ljpeg -lm -lmpfit


CC=g++

OFILES=mpfit.o
LIBFILE=libmpfit.a

all: $(LIBFILE) idscheck 	

$(LIBFILE): $(OFILES)
    $(AR) r $@ $(OFILES)

mpfit.o: mpfit.c mpfit.h
        $(CC) -c -o $@ $< $(CFLAGS)

$(LIBFILE): $(OFILES)
        $(AR) r $@ $(OFILES)	
clean:
	rm -f *.o 

idscheck.o: idscheck.cpp
	$(CC) $(CCDEBUG) $(CCINCS) -o $@ -c $< $(CFLAG)

idscheck: idscheck.o
	$(CC) -o $@ $< $(CFLAG) $(CCLIBS) $(CCLINK)

	
pushprocyon:
	rsync -avuz /Users/chyan/Documents/workspace/IDSControl/* -e ssh chyan@procyon:/home/chyan/Development/idsexposure
	
pushspica:
	rsync -avuz ../IDSControl -e ssh chyan@spica:/home/chyan/
