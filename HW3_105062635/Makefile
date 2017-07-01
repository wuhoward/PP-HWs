CC			:= gcc
CXX			:= g++

MPICC			:= mpiicc
MPICXX			:= mpic++

CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
CXXFLAGS		:= -O3 -march=native -Wall -std=c++0x
LDFLAGS			:= -lX11 -pthread 

all: SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async

SSSP_Pthread: SSSP_Pthread.c
	$(CC) $(CCFLAGS) $(LDFLAGS) -o $@ $?
SSSP_MPI_sync: SSSP_MPI_sync.c
	$(MPICC) $(CCFLAGS) $(LDFLAGS) -o $@ $?
SSSP_MPI_async: SSSP_MPI_async.c
	$(MPICC) $(CCFLAGS) $(LDFLAGS) -o $@ $?

clean:
	rm -f SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async
