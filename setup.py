import os

def home_dir():
    noteshomedir = '/global/u1/z/zzhang13/SPT_DES_Clusters/'
    if not os.path.exists(noteshomedir):
        raise Exception('something is very wrong: %s does not exist'%noteshomedir)
    return noteshomedir

def code_home_dir():
    codehomedir = os.path.join(home_dir(), 'code/')
    return codehomedir

def output_home_dir():
    outphomedir = os.path.join(home_dir(), 'output/')
    return outphomedir

def data_home_dir():
    datahomedir = os.path.join(home_dir(), 'data/')
    if not os.path.exists(datahomedir):
        os.makedirs(datahomedir)
    return datahomedir

def bigdata_home_dir():
    datahomedir = '/global/cscratch1/sd/zzhang13/'
    if not os.path.exists(datahomedir):
        os.makedirs(datahomedir)
    return datahomedir



