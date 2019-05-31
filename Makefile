CC = g++
CFLAGS = -g -std=c++11 -Wall
SRCS = main.cpp
PROG = seamCarve.out

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
clean:
	rm *.out
