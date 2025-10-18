import os

from fastsafetensors import cpp

def main():
    # for cpp debug log
    os.environ['FASTSAFETENSORS_ENABLE_INIT_LOG'] = '1'

    if cpp.is_cufile_found():
        print(cpp.cufile_version())


if __name__ == "__main__":
    main()
