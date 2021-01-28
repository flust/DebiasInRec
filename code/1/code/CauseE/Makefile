CXX = g++
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = g++
endif

#To init MKLROOT just Do  "dot white dot/init.sh". That is, ". ./init.sh" 
#MKLROOT = /home/ybw/intel/compilers_and_libraries_2018.2.199/linux/mkl

CXXFLAGS = -Wall -O3 -std=c++0x -march=native
MKLFLAGS = -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
DFLAG += -DUSEOMP
#DFLAG += -DEBUG
CXXFLAGS += -fopenmp
CXXFLAGS += $(MKLFLAGS)

all: train


train: train.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f train predict ffm.o *.bin.*
