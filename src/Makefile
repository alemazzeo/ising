.PHONY: default help object executable all clean
CC = gcc

LD = $(CC)

BIN = ../bin/

SOURCE_C = $(wildcard *.c)
OBJECTS_C = $(patsubst %.c, $(BIN)%_c.o, $(SOURCE_C))

CFLAGS = -std=gnu99 -Wall
EXECUTABLE = ../bin/ising.e
LDFLAGS = -lm

default: help

help:
	@echo "\
Options:\n\n\
  make objects:       compiler makes objects for every *.c\n\
  make executable:    compiler makes executable\n\
  make all:           build all previous\n\
  make clean:         delete output files\n\
  make help:          display this help"

objects: $(OBJECTS_C)

executable: $(EXECUTABLE)

all: objects executable

$(BIN)%_c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.e: $(OBJECTS_C)
	$(LD) $^ $(LDFLAGS) -o $@

clean:
	rm -rfv $(OBJECTS_C) $(EXECUTABLE)
