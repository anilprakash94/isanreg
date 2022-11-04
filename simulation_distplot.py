import argparse
import os
import errno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def distance_data(attr_output):
    with open(attr_output,"r") as f:
        file_data = f.readlines()
    file_data = file_data[1:]
    i=0
    dist_list = []
    while i < len(file_data):
        if file_data[i].split()[0] == file_data[i+1].split()[0]:
            m1 = file_data[i].split()[2]
            m1 = m1.split(":")
            m1 = [float(x) for x in m1]
            m2 = file_data[i+1].split()[2]
            m2 = m2.split(":")
            m2 = [int(x) for x in m2]
            if m1[1] < m2[0]:
                dist = m2[0] - m1[1]
            else:
                dist = m1[1] - m2[0]
            dist_list.append(dist)
            i += 2
        else:
            i += 1
    return dist_list

def plot_dist(dist_list,args):
    dist_list = np.array(dist_list)
    f = plt.figure()
    f.set_figwidth(25)
    f.set_figheight(15)
    sns.swarmplot(x=dist_list,color='cornflowerblue', size=7)
    plt.xlabel('Distance between motifs (bp)', labelpad=15,fontsize=25,weight='bold')
    plt.xticks(fontsize=21)
    plot_name = args.image_dir + "/distance_plot.png"
    plt.savefig(plot_name,dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate swarm plot')
    parser.add_argument('-attr', '--attr_output', default="Models/simulated/simtest_out.txt", help='specify output text file having attribution results')
    parser.add_argument('--image_dir', default="Models/simulated", help='specify folder location for saving swarm plot')
    args, unknown = parser.parse_known_args()
    
    #raises exception if the input file does not exist
    
    if not os.path.isfile(args.attr_output):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),args.attr_output)
    
    #calculating distance between motifs in bp
    dist_list = distance_data(args.attr_output)
    
    #plotting distance values
    plot_dist(dist_list,args)
    

