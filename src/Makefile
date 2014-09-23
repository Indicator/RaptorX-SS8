mpitrain:
	mpiCC -o bcnf_mpitp.train bCNF_mpi_tp.cpp -O3 -D_MPI
	cp bcnf_mpitp.train ../bin
train:
	g++ -o bcnf_mpitp.train bCNF_mpi_tp.cpp -O3 
	cp bcnf_mpitp.train ../bin
predict:
	g++ -o bcnf_mpitp bCNF_mpi_tp.cpp -O3
	cp bcnf_mpitp ../bin

install:
	cp bcnf_mpitp ../bin
	cp bcnf_mpitp.train ../bin
test:
	make train
	make predict
	../bin/test.sh

debug:
	g++ -o bcnf_mpitp.train bCNF_mpi_tp.cpp -g
