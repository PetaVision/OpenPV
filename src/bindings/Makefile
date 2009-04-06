all: pypv

pv_mex:
	mkoctfile --mex pv_mex.cpp -L.. -lpv
	mv -f pv_mex.mex ..
	#mv -f pv_mex.mexglx ..

pypv:
	python setup.py build
	cp build/lib.linux-i686-2.5/pypv.so .
	cp build/lib.linux-i686-2.5/pypv.so ..

clean:
	rm -f pv_mex.mex
	rm -f *.so ../*.so
	rm -f build/*/*
