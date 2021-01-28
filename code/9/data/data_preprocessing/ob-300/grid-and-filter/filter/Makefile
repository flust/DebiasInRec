CXX = g++
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = g++
endif

#To init MKLROOT just Do  "dot white dot/init.sh". That is, ". ./init.sh" 
#MKLROOT = /home/ybw/intel/compilers_and_libraries_2018.2.199/linux/mkl

CXXFLAGS = -Wall -O3 -std=c++0x -march=native


#Comment out the following two sentences if MKL is required
DFLAG += -DMKL
BLASFLAGS = -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

#DFLAG += -DOPENBLAS
#BLASFLAGS =  -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas -lpthread

#DFLAG += -DUSEOMP
#DFLAG += -DEBUG
#DFLAG += -D EBUG_nDCG
#DFLAG += -D SHOW_SCORE_ONLY
#DFLAG += -D DEBUG_SAVE
CXXFLAGS += -fopenmp

all: predict-and-filter


predict-and-filter: predict-and-filter.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BLASFLAGS)
ffm.o: ffm.cpp ffm.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $< $(BLASFLAGS)

clean:
	rm -f predict-and-filter predict ffm.o *.bin.*
