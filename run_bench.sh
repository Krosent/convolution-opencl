
for j in {1..30}
    do
        echo "> Basic Benchmark has been started, $j iteration"
    	PYOPENCL_COMPILER_OUTPUT=1 python basic.py sonic.jpg >> output-bench-basic.txt
done

#for j in {1..30}#
#   do
#        echo "> OpenCL Benchmark has been started, $j iteration"
#    	PYOPENCL_COMPILER_OUTPUT=1 python cl_optimized.py sonic.jpg >> output-bench-opencl.txt
#done