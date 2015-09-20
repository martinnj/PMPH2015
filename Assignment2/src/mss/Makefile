CXX        = nvcc

SOURCES_CPP =mssMain.cu
HELPERS     =ScanHost.cu.h ScanKernels.cu.h
EXECUTABLE  =mssMain


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
