CXX        = nvcc

SOURCES_CPP =task3.cu
HELPERS     =cpuFunc.cu.h gpuFunc.cu.h
EXECUTABLE  =task3


default: clean compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) -arch=compute_20 -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
