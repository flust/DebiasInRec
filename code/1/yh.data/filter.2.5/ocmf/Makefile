CXX = g++
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = g++-6
endif

#MKLPATH = /home/skypole/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin
#MKLINCLUDE = /home/skypole/intel/compilers_and_libraries_2018.2.199/linux/mkl/include
MKLROOT = /home/skypole/intel/compilers_and_libraries_2018.2.199/linux/mkl
CXXFLAGS = -Wall -O3 -std=c++0x -march=native
MKLFLAGS = -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
#DFLAG += -D DEBUG
CXXFLAGS += -fopenmp
#CXXAGS += $(MKLFLAGS)

all: train predict

predict: predict.cpp mf.o
	$(CXX) $(CXXFLAGS) -o $@ $^

train: train.cpp mf.o
	$(CXX) $(CXXFLAGS)  $(DFLAG) -o $@ $^

mf.o: mf.cpp mf.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f train predict mf.o *.bin.*
