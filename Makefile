
OPENCVINC = `pkg-config --cflags opencv`
OPENCVLNK = `pkg-config --libs opencv`

CCDEBUG = -g -Ofast -Wall -Wno-write-strings
CCINCS = -I/usr/include -I. -I/usr/local/include $(OPENCVINC)
CCLIBS = -L/usr/lib/x86_64-linux-gnu -L. -L/usr/local/lib $(OPENCVLNK)
CCLINK = -ljpeg -lcfitsio -lm -lueye_api


CC=g++


all: idsexposure

clean:
	rm -f *.o 

idsexposure.o: idsexposure.c
	$(CC) $(CCDEBUG) $(CCINCS) -o $@ -c $< $(CFLAG)

idsexposure: idsexposure.o
	$(CC) -o $@ $< $(CFLAG) $(CCLIBS) $(CCLINK)

	
pushprocyon:
	rsync -avuz /Users/chyan/Documents/workspace/IDSControl/* -e ssh chyan@procyon:/home/chyan/Development/idsexposure
	
pushspica:
	rsync -avuz /Users/chyan/Documents/workspace/IDSControl -e ssh chyan@spica:/home/chyan/