CP=g++
CPFLAGS=-std=c++11 -pthread -pedantic -Wall
CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(CPP_FILES:.cpp=.o)

all: main

debug: CPFLAGS += -DDEBUG -g
debug: main

main: main.o ivector.o vector.o layer.o neuron.o backprop.o 
	$(CP) $(CPFLAGS) $^ -o $@ -lm

%.o:%.cpp
	$(CP) $(CPFLAGS) -c $< -o $@ -lm

clean:
	$(RM) main $(OBJ_FILES) *.h.gch

zip:
	zip -r xprofa00.zip *.cpp *.h Makefile run.sh data/ data_l2-norm list.txt xprofa00.pdf
