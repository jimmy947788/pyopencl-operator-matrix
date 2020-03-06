import pyopencl as cl

if __name__ == "__main__":
    
    for platfrom in cl.get_platforms():
        platfrom_name = platfrom.get_info(cl.platform_info.NAME)
        platfrom_extenstions = platfrom.get_info(cl.platform_info.EXTENSIONS)
        print("%s :%s" % (platfrom_name, platfrom_extenstions))

    for device in platfrom.get_devices(cl.device_type.GPU):
        print("---------------------------------------------------------------")
        print("    Device name:", device.name)
        print("    Device type:", cl.device_type.to_string(device.type))  # @UndefinedVariable
        print("    Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print("    Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("    Device compute units:", device.max_compute_units)
        print("    Device max work items:", device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))  # @UndefinedVariable
        print("    Device local memory:", device.get_info(cl.device_info.LOCAL_MEM_SIZE)//1024, 'KB')  # @UndefinedVariable 
        print("---------------------------------------------------------------")