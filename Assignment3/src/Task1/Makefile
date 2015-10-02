CXX        = nvcc

SOURCES_CPP =task1.cu
HELPERS     =cpuFunc.cu.h gpuFunc.cu.h
EXECUTABLE  =task1


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
