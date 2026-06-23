# Nomi eseguibili
FACE = main

# Compilatori
CXX := g++
CC  := gcc 

# Directory
SRC_DIR = src
OBJ_DIR = objects

# Flags
CFLAGS := -Wall -Wpointer-arith -std=c++11 -ffast-math
LDFLAGS := $(shell pkg-config --libs opencv) -lpthread -lhineon -ln2cube -ldputils

# Modelli e Architettura
CUR_DIR = $(shell pwd)
MODEL = $(CUR_DIR)/models/recognizer/dpu_face_resnet50.elf
ARCH := $(shell uname -m | sed -e s/arm.*/armv7l/ -e s/aarch64.*/aarch64/)

ifeq (debug, $(filter debug,$(MAKECMDGOALS)))
	CFLAGS += -g
else
	CFLAGS += -O2
endif

ifeq ($(ARCH),armv7l)
    CFLAGS += -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon -mno-unaligned-access
endif
ifeq ($(ARCH),aarch64)
    CFLAGS += -mcpu=cortex-a53
endif


COMMON_SRC := $(filter-out $(SRC_DIR)/main.cpp $(SRC_DIR)/enroll.cpp, $(wildcard $(SRC_DIR)/*.cpp))
COMMON_OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(COMMON_SRC))

# Oggetti specifici
FACE_OBJ := $(OBJ_DIR)/main.o $(COMMON_OBJ)

.PHONY: all debug clean

all: $(FACE) $(ENROLL)
debug: all

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c $(CFLAGS) $< -o $@

$(FACE): $(FACE_OBJ)
	$(CXX) $(CFLAGS) $^ $(MODEL) -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(FACE)