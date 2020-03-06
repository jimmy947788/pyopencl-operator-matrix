import pyopencl as cl

if __name__ == "__main__":
    
    for platfrom in cl.get_platforms():
        platfrom_name = platfrom.get_info(cl.platform_info.NAME)
        platfrom_extenstions = platfrom.get_info(cl.platform_info.EXTENSIONS)
        print("%s :%s" % (platfrom_name, platfrom_extenstions))