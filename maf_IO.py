from sys import stdout
import numpy as np
import warnings
import os
import csv
#stdout.write("\r{0}/{1}".format(n,n_samp))
#stdout.flush()

def downsample(array, min_sep, tol=0.10):
    n=0
    samples=[n]
    while array[samples[-1]]!=array[-1]:
        pio_index = samples[n-1]
        pioneer = array[pio_index]
        for i in xrange(pio_index+1,len(array)):
            sep = array[i]-pioneer
            if sep<=0:
                raise RuntimeError, "Null or negative separation,\nencountered between position {0} and {1}".format(pio_index,i)
            elif sep>min_sep*(1+tol):
                samples.append(i)
                break
            elif min_sep*(1-tol) < sep < min_sep*(1+tol):
                samples.append(i)
                break
            elif array[-1]-pioneer<=min_sep:
                samples.append(len(array)-1)
                break
            elif array[samples[-1]]==array[-1]:
                break
            else:
                continue
        else:
            warnings.warn("\nCould not perform appropiate sampling over given array.\nTolerance = {0}%\n . . . Reverting to a full sample.".format(100*tol)
                          ,RuntimeWarning)
            # use everything as sample space and set samp_rate as None
            samples = range(len(array))
            break
    return samples

def outwrite(space, data, samp_steps=[1e-4,None,None], fprefix="eggs", wdir=".", tol=0.10):
    """
    This function takes as input space, a list of 1-d arrays, and data, an n-d array.
    The list of space arrays has to coincide with the order in which this dimensions are represented in the data array.
    This function writes our data and space coordinates into two files;
    one with suffix "_data.csv",
    and the other with "_space.csv".
    This function does not return anything.
    """
    ################################################################################################
    # pre-process
    
    num_dim = len(space)
    num_datasets = data.shape[0]
    files = ["_data.csv","_space.csv"]

    # catch errors
    if type(data)!=np.ndarray:
        raise ValueError, "Data is not a numpy.ndarray."
    if data.ndim!=1+num_dim:
        raise ValueError, "Shape of data and space arrays do not match.\nSpace has have {1} dimensions, data lives in {0}, mind you.".format(data.ndim-1,num_dim)
    else:
        space_params = {i:{} for i in xrange(num_dim)}
        for i in xrange(len(space)):
            if type(space[i])!=np.ndarray:
                raise ValueError, "Dimension {0} in not a numpy.ndarray.".format(i)
            elif len(space[i].shape)!=1:
                raise ValueError, "Invalid shape of dimension {0} array.".format(i)
            else:
                # set the stage
                # with some parameters of total available data
                space_params[i]["len"] = len(space[i])
                # minimum separation between samples
                if samp_steps[i]:
                    # construct sample space
                    space_params[i]["samples"] = downsample(space[i],samp_steps[i],tol=tol)
                else:
                    space_params[i]["samples"] = range(space_params[i]["len"])


    ################################################################################################
    # Create and write the goddamn files

    # SPACE FILE
    # the space file is special if there is downsampling
    # if I detect a previous file, I'll assume we are carrying on with that
    if fprefix+"_space.csv" not in os.listdir(wdir):
        # create
        spacefile = (open(wdir+"/"+fprefix+"_space.csv","wb"))
        print "\t. . . creating new space file"
        # write
        spacewriter = csv.writer(spacefile, delimiter=",")
        for n in xrange(num_dim):
            spacewriter.writerow(space[n][space_params[n]["samples"]])
        spacefile.close()
    else:
        print "\t. . . space file already exists"
        # read
        old_spacefile = open(wdir+"/"+fprefix+"_space.csv","rb")
        spacereader = csv.reader(old_spacefile, delimiter=",")
        new_space = []
        # compare
        space_flag = 0
        for n in xrange(num_dim):
            old_dim = np.array(spacereader.next(),dtype=np.float32)
            new_dim = space[n][space_params[n]["samples"]]
            if len(old_dim)==len(new_dim) and np.all(np.isclose(new_dim.astype(np.float32),old_dim,atol=1e-16)):
                new_space.append(old_dim)
                space_flag += 1
            elif len(np.where(new_dim.astype(np.float32)==old_dim[-1])[0])==1:
                start_point = np.where(new_dim.astype(np.float32)==old_dim[-1])[0][0]+1
                # aqui, cambiar
                new_space.append(np.concatenate( (old_dim,new_dim[start_point:]) ))
            else:
                raise RuntimeError, "Dimension {0} did not match, or could not be glued.".format(n)
        # create
        spacefile = (open(wdir+"/"+fprefix+"_space.csv","wb"))
        # write
        spacewriter = csv.writer(spacefile, delimiter=",")
        for n in xrange(num_dim):
            spacewriter.writerow(new_space[n])
        spacefile.close()
        old_spacefile.close()


    # DATA FILE
    # create
    if fprefix+"_data.csv" not in os.listdir(wdir):
        datafile = open(wdir+"/"+fprefix+"_data.csv","wb")
        print "\t. . . creating new data file"
    elif space_flag == num_dim and fprefix+"_data.csv" in os.listdir(wdir):
        print "\t. . . data file already exists, but space is unchanged; nothing written"
        return None
    else:
        datafile = open(wdir+"/"+fprefix+"_data.csv","ab")
        print "\t. . . data file already exists, appending new data"
    datawriter = csv.writer(datafile, delimiter=",")
    # write
    # esto es super chaca, podria ser un iterator
    for t in space_params[0]["samples"]:
        for y in space_params[1]["samples"]:
            for x in space_params[2]["samples"]:
                datawriter.writerow(data[:,t,y,x])
    datafile.close()

def read_space(prefix = "eggs", rdir = "."):
    space = []
    spacefile = open(rdir+"/"+prefix+"_space.csv","r")
    spacereader = csv.reader(spacefile,delimiter=",",)
    for n in spacereader:
        space.append(np.array(n,dtype=np.float32))
    return space

def read(prefix = "eggs", rdir = "."):
    space = read_space(prefix=prefix, rdir=rdir)
    datafile = open(rdir+"/"+prefix+"_data.csv")
    n_data = len(datafile.readline().split(","))
    n_space = len(space)
    datafile.seek(0)
    datareader = csv.reader(datafile,delimiter=",")
    datamatrix = np.empty([n_data]+[len(d) for d in space])
    for t in xrange(len(space[0])):
        for y in xrange(len(space[1])):
            for x in xrange(len(space[2])):
                datamatrix[:,t,y,x] = np.array(datareader.next(),dtype=np.float32)
    return datamatrix, space
