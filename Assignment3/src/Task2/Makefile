CXX        = nvcc

SOURCES_CPP =task2.cu
HELPERS     =cpuFunc.cu.h gpuFunc.cu.h
EXECUTABLE  =task2


default: clean compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<

bonus: clean task2openMP
task2openMP:
	g++ -fopenmp -o task2openMP task2openMP.cpp

compile: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) -arch=compute_20 -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f task2openMP
