CXX        = nvcc
CXXFLAGS   = -arch=compute_20

SOURCES_CPP =task13.cu
HELPERS     =cpuFunc.cu.h gpuFunc.cu.h ScanHost.cu.h ScanKernels.cu.h
EXECUTABLE  =task13


default: clean compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
