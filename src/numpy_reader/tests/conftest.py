import os
import pytest

#additional test arguments
def pytest_addoption(parser):
    parser.addoption("--device", type=int, default=-1, help="specify the device to test (CPU: -1, GPU: 0:NUM_GPU-1)")
    parser.addoption("--filepath", type=str, default="/data/tests", help="path to the test files")


@pytest.fixture
def devopt(request):
    return request.config.getoption("--device")


@pytest.fixture
def filepathopt(request):
    return request.config.getoption("--filepath")


#generate list of filenames
def pytest_generate_tests(metafunc):
    if "filename_row_major" in metafunc.fixturenames:
        path = metafunc.config.getoption("filepath")
        filenames = [os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".npy") and "_rm" in x and "_fail" not in x)]
        metafunc.parametrize("filename_row_major", filenames)
    
    if "filename_column_major" in metafunc.fixturenames:
        path = metafunc.config.getoption("filepath")
        filenames = [os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".npy") and "_cm" in x and "_fail" not in x)]
        metafunc.parametrize("filename_column_major", filenames)
    
    if "filename_all" in metafunc.fixturenames:
        path = metafunc.config.getoption("filepath")
        filenames = [os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".npy") and "_fail" not in x)]
        metafunc.parametrize("filename_all", filenames)



#additional info
def pytest_report_header(config):
    devid = config.getoption("device")
    fpath = config.getoption("filepath")
    if (devid == -1):
        return ["testing IO for CPU from path {}".format(fpath)]
    else:
        return ["testing IO for GPU{} from path {}".format(devid, fpath)]

