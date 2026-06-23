CXX := g++
EXE := cvinfo

CFLAGS := -O2 -Wall -std=c++11 $(shell pkg-config --cflags opencv)

LIBS := $(shell pkg-config --libs opencv)

SRC_FILES = main.cpp

$(EXE): $(SRC_FILES)
	$(CXX) $(SRC_FILES) $(CFLAGS) -o $(EXE) $(LIBS)

clean:
	rm -f $(EXE)