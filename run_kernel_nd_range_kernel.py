import pyopencl as cl
import numpy as np

if __name__ == "__main__":
    
    # Create context and command queue
    platform = cl.get_platforms()[0]
    devices = platform.get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    # Open program file and build
    program_file = open('kernels/vector_multi_matrix.cl', 'r') # Scale x Matrx
    program_text = program_file.read()
    program = cl.Program(context, program_text)
    try:
        program.build()
    except:
        print("Build log:")
        print(program.get_build_info(devices[0], 
                cl.program_build_info.LOG))
        raise

    matrix_width = 9
    matrix_height = 20
    # Create arguments for kernel: a scalar, a LocalMemory object, and a buffer
    vector = np.random.randint(2, size=matrix_width).astype(np.float32)
    print("vector=", vector)
    marix = np.random.randint(0, 1500, size=(matrix_width * matrix_height)).astype(np.float32)
    print("marix=", marix)

    # create buffer READ/WRITE  cl.mem_flags.READ_WRITE
    buffer_vector = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)
    buffer_matrix = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=marix)
    buffer_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, vector.nbytes)
    #float_row_width = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=(matrix_width, ))

    # Create, configure, and execute kernel (Seems too easy, doesn't it?)
    global_work_offset = (0, )
    global_work_size = (matrix_width, )
    local_work_size = (1, )
    kernel = program.vector_multi_matrix #(queue, (25,), (25,), scalar, float_buffer, lm)
    kernel.set_arg(0, buffer_vector)
    kernel.set_arg(1, buffer_matrix)
    kernel.set_arg(2, buffer_result)
    kernel.set_arg(3, np.int32(matrix_height))

    ev = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size, global_work_offset)

    # Read data back from buffer
    result = np.empty_like(vector)
    cl.enqueue_read_buffer(queue, buffer_result, result).wait()

    print("result=", result)

