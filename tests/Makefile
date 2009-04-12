SRCDIR = ../src
BUILDDIR = build

include $(SRCDIR)/include/sources.mk

INPARAMS = input/inparams.txt

#CPP = mpic++
#CC  = mpicc
CPP = c++
CC  = cc

MPI_DIR = ../src/arch/mpi
MPI_INC = -I$(MPI_DIR)
MPI_OBJ = $(MPI_DIR)/mpi.o

CFLAGS = -g -std=c99 -Wall $(MPI_INC)
CPPFLAGS = -g -Wall $(MPI_INC)

DEPENDS = $(HEADERS)

TESTS = test_kg test_kxpos test_kypos \
        test_pos test_delta_pos test_extend_border \
        test_patch_head test_delta test_sign test_decode_msg \
        test_borders test_gs_io test_layer_direct

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

$(BUILDDIR)/%.o: %.cpp
	@echo 'Building file: $<'
	$(CPP) $(CPPFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

$(BUILDDIR)/%.o: %.c
	@echo 'Building file: $<'
	$(CC) $(CFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: make-in-subdirs $(TESTS)

test_layer_direct: $(BUILDDIR)/test_layer_direct.o $(BUILDDIR)/TestLayer.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $(BUILDDIR)/$@.o $(BUILDDIR)/TestLayer.o $(OBJS) $(MPI_OBJ)

test_gs_io: $(BUILDDIR)/test_gs_io.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $(BUILDDIR)/$@.o $(OBJS) $(MPI_OBJ)

test_kg: $(BUILDDIR)/test_kg.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_kxpos: $(BUILDDIR)/test_kxpos.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_kypos: $(BUILDDIR)/test_kypos.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_delta_pos: $(BUILDDIR)/test_delta_pos.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_pos: $(BUILDDIR)/test_pos.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_decode_msg: $(BUILDDIR)/test_decode_msg.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_extend_border: $(BUILDDIR)/test_extend_border.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_borders: $(BUILDDIR)/test_borders.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $(BUILDDIR)/$@.o $(OBJS) $(MPI_OBJ)

test_patch_head: $(BUILDDIR)/test_patch_head.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $(BUILDDIR)/$@.o $(OBJS) $(MPI_OBJ)

test_delta: $(BUILDDIR)/test_delta.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

test_sign: $(BUILDDIR)/test_sign.o $(DEPENDS)
	$(CC) -o $@ $(BUILDDIR)/$@.o

$(MPI_OBJ): $(MPI_DIR)/mpi.h $(MPI_DIR)/mpi.c
	$(CC) $(CFLAGS) -c $(MPI_DIR)/mpi.c -o $@

runtests: make-in-subdirs $(TESTS)
	@rm -rf output
	@mkdir output
	@echo "Running test_kg ..."
	@test_kg
	@echo "Running test_kxpos ..."
	@test_kxpos
	@echo "Running test_kypos ..."
	@test_kypos
	@echo "Running test_pos ..."
	@test_pos
	@echo "Running test_delta_pos ..."
	@test_delta_pos
	@echo "Running test_extend_border ..."
	@test_extend_border
	@echo "Running test_patch_head ..."
	@test_patch_head
	@echo "Running test_delta ..."
	@test_delta
	@echo "Running test_sign ..."
	@test_sign
	@echo "Running test_decode_msg ..."
	@test_decode_msg
	@echo "Running test_borders ..."
	@test_borders -i 1 -p $(INPARAMS)
	@echo "Running test_gs_io with 1 node ..."
	@test_gs_io -i 1 -p $(INPARAMS)
	@echo "Running test_gs_io with 2 nodes ..."
	@mpirun -np 2 test_gs_io -i 1 -p $(INPARAMS)
	@echo "Running test_gs_io with 4 nodes ..."
	@mpirun -np 4 test_gs_io -i 1 -p $(INPARAMS)
	@echo "Running test_layer_direct with 1 node ..."
	@test_layer_direct -i 1 -p $(INPARAMS)

make-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "making in $$subdir" ; \
	        (cd $$subdir && $(MAKE) all) ; \
	    fi \
	done

clean: clean-in-subdirs
	rm -f $(TESTS) *.o *.bin
	rm -f $(BUILDDIR)/*.o

cleanall: clean
	rm -f $(MPI_OBJ)
	rm -rf output

clean-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "cleaning in $$subdir" ; \
	        (cd $$subdir && $(MAKE) clean) ; \
	    fi \
	done

force:

include $(SRCDIR)/include/depend.mk