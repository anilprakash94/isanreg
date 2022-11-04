
import os
import argparse
import math

def weights_filepath(model_dir):
    current_dir = os.getcwd()
    dir_path = os.path.join(current_dir,model_dir)
    if os.path.isdir(dir_path):
        pass   
    else:
        os.makedirs(dir_path)
    return dir_path


def circos_inputs(args):
    win_len = args.window_len
    dir_path = weights_filepath(args.data_dir + "/" + "circos_" + args.tf_name)
    with open(args.out_file, "r") as f:
        file = f.readlines()
    file = file[1:101]
    label_file = dir_path + "/" + "labels.txt"
    #writes text label file
    with open(label_file, "w") as f:
        idx = 0
        for i in file:
            line = i.strip().split()
            f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\t" + line[0] + "\n")
            idx += win_len
        end = idx
    pval_plot = dir_path + "/" + "pval.txt"
    #writes p-value scatter plot file
    with open(pval_plot, "w") as f:
        idx = 0
        for i in file:
            line = i.strip().split()
            pval = float(line[2])
            if pval == 0:
                nlog_pval = -math.log10(1e-300)
            else:
                nlog_pval = -math.log10(pval)
            f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\t" + str(nlog_pval) + "\n")
            idx += win_len
    adjp_plot = dir_path + "/" + "adjpval.txt"
    #writes adjusted p-value file
    with open(adjp_plot, "w") as f:
        idx = 0
        for i in file:
            line = i.strip().split()
            adj_pval = float(line[3])
            if adj_pval == 0:
                nlog_adj = -math.log10(1e-300)
            else:
                nlog_adj = -math.log10(adj_pval)
            f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\t" + str(nlog_adj) + "\n")
            idx += win_len
    odds_plot = dir_path + "/" + "oddsratio.txt"
    #writes odds ratio heat map file
    with open(odds_plot, "w") as f:
        idx = 0
        for i in file:
            line = i.strip().split()
            f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\t" + line[1] + "\n")
            idx += win_len
    karyotype_plot = dir_path + "/" + "karyotype.txt"
    #writes karyotype input file
    with open(karyotype_plot, "w") as f:
        f.write("chr" + "\t" + "-" + "\t" + "axis1" + "\t" +  "1" + "\t" + "0" + "\t" + str(end) + "\t" + "blue" + "\n")
    hilt_plot = dir_path + "/" + "highlights.txt"
    #writes highlight file
    with open(hilt_plot, "w") as f:
        idx = 0
        for i in file:
            line = i.strip().split()
            adj_pval = float(line[3])
            if adj_pval < 0.05:
                f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\n")
            idx += win_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Circos input generation')
    parser.add_argument('--data_dir', default="data_files", help='name of folder having data files')
    parser.add_argument('--window_len', type=int, default=5, help='specify the length of sliding window')
    parser.add_argument('--tf_name', default = "ESR1", help='name of TF under study')
    parser.add_argument('--out_file', default = "Models/ESR1/ESR1_out.txt", help='testing output file')
    args, unknown = parser.parse_known_args()
    
    #writes circos input files
    circos_inputs(args)
    
