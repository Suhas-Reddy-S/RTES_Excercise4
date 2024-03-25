INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -lpthread

HFILES= 
CFILES= Q5a.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	Q5a

clean: 
	-rm -f *.o *.d
	-rm -f Q5a

Q5a: Q5a.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

depend:

.cpp.o: 
	$(CC) $(CFLAGS) -c $<
