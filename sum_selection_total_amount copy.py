import pyopencl as cl
import numpy as np
import itertools
import time

if __name__ == "__main__":
    
    # Create context and command queue
    platform = cl.get_platforms()[0]
    devices = platform.get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    # Open program file and build
    program_file = open('kernels/sum_selection_total_amount.cl', 'r') # Scale x Matrx
    program_text = program_file.read()
    program = cl.Program(context, program_text)
    try:
        program.build()
    except:
        print("Build log:")
        print(program.get_build_info(devices[0], 
                cl.program_build_info.LOG))
        raise

    selection_length = 10
    wager_length = 100
    print("matrix size =", selection_length, "x",  wager_length)
    # Create arguments for kernel: a scalar, a LocalMemory object, and a buffer
    selection = np.random.randint(2, size=selection_length).astype(np.float32)
    print("selection=", selection)

    wagers = np.random.randint(0, 1500, size=(selection_length * wager_length)).astype(np.float32)
    print("marix=", wagers)

    tStart = time.time()#計時開始
    # create buffer READ/WRITE  cl.mem_flags.READ_WRITE
    buffer_selection = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=selection)
    buffer_wagers = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=wagers)
    buffer_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, wager_length)

    # Create, configure, and execute kernel (Seems too easy, doesn't it?)
    global_work_offset = (0, )
    global_work_size = (selection_length, )
    local_work_size = (1, )
    kernel = program.sum_selection_total_amount
    kernel.set_arg(0, buffer_selection)
    kernel.set_arg(1, buffer_wagers)
    kernel.set_arg(2, buffer_result)
    kernel.set_arg(3, np.int32(wager_length))

    ev = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size, global_work_offset)

    # Read data back from buffer
    result = np.empty(selection_length, dtype=np.float32)
    cl.enqueue_read_buffer(queue, buffer_result, result).wait()
    tEnd = time.time()#計時結束
    print("It cost %f sec" % (tEnd - tStart))#會自動做近位

    print("result=", result)

