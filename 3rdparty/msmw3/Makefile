CXXFLAGS ?= -march=native -O3
override LDLIBS  := $(LDLIBS) -lfftw3f -ltiff -lpng

OBJ = main.o \
      LibImages/LibImages.o \
      LibMSMW/ConnectedComponents.o \
      LibMSMW/LibMSMW.o \
      LibMSMW/UtilitiesMSMW.o \
      Utilities/Memory.o \
      Utilities/Parameters.o \
      Utilities/Time.o \
      Utilities/Utilities.o

msmw: $(OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f msmw $(OBJ)
