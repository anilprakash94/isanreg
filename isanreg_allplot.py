import os
import argparse
import math
import pandas as pd

#loading all testing output files in the folder

def weights_filepath(model_dir):
    current_dir = os.getcwd()
    dir_path = os.path.join(current_dir,model_dir)
    if os.path.isdir(dir_path):
        pass   
    else:
        os.makedirs(dir_path)
    return dir_path


def extract_files(dataset_path):
    test_files = []
    file_names = os.listdir(dataset_path)
    for names in file_names:
        if names.endswith("out.txt"):
            test_files.append(names)
    return test_files


def extract_res(test_files,dataset_path,top_num):
    for path in test_files:
        filepath = dataset_path + "/" + path
        tf_name = path.split("_")[0]
        file_df = pd.read_table(filepath,header=0,index_col=False)
        file_df = file_df.sort_values(by=['corrected_p_value'])
        file_df = file_df.head(top_num)
        file_df = file_df.assign(tf=tf_name)
        if path == test_files[0]:
            test_df = file_df
        else:
            test_df = test_df.append(file_df, ignore_index = True)
    return test_df


def circos_all(test_df,dataset_path,args,top_num):
    win_len = args.window_len
    dir_path = weights_filepath(dataset_path + "/" + "circos_all")
    label_file = dir_path + "/" + "labels.txt"
    #writes text label file
    tf_dict = {}
    outer_label = []
    outer_tile = []
    with open(label_file, "w") as f:
        idx = 0
        for i in range(0, len(test_df)):
            kmer_seq = test_df.iloc[i]['k-mer']
            tf_col = test_df.iloc[i]['tf']
            if tf_col not in tf_dict:
                tf_dict[tf_col] = 0
                tf_idx = idx
            tf_dict[tf_col] += 1
            f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\t" + kmer_seq + "\n")
            if tf_dict[tf_col] >= top_num:
                outer_label.append("axis1" + "\t" + str(tf_idx) + "\t" + str(idx+4) + "\t" + tf_col + "\n")
                outer_tile.append("axis1" + "\t" + str(tf_idx) + "\t" + str(idx+4) + "\n")
            idx += win_len
        end = idx
    #writes outer label file
    outer_plot = dir_path + "/" + "out_label.txt"
    with open(outer_plot, "w") as f:
        for i in outer_label:
            f.write(i)
    #writes outer tile file
    out_tile = dir_path + "/" + "out_tile.txt"
    with open(out_tile, "w") as f:
        for i in outer_tile:
            f.write(i)
    pval_plot = dir_path + "/" + "pval.txt"
    #writes p-value histogram file
    with open(pval_plot, "w") as f:
        idx = 0
        for i in range(0, len(test_df)):
            pval = float(test_df.iloc[i]['p_value'])
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
        for i in range(0, len(test_df)):
            adj_pval = float(test_df.iloc[i]['corrected_p_value'])
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
        for i in range(0, len(test_df)):
            odds = test_df.iloc[i]['odds_ratio']
            f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\t" + str(odds) + "\n")
            idx += win_len
    karyotype_plot = dir_path + "/" + "karyotype.txt"
    #writes karyotype input file
    with open(karyotype_plot, "w") as f:
        f.write("chr" + "\t" + "-" + "\t" + "axis1" + "\t" +  "1" + "\t" + "0" + "\t" + str(end) + "\t" + "blue" + "\n")
    hilt_plot = dir_path + "/" + "highlights.txt"
    #writes highlight file
    with open(hilt_plot, "w") as f:
        idx = 0
        for i in range(0, len(test_df)):
            adj_pval = float(test_df.iloc[i]['corrected_p_value'])
            adj_pval = float(adj_pval)
            if adj_pval < 0.05:
                f.write("axis1" + "\t" + str(idx) + "\t" + str(idx+4) + "\n")
            idx += win_len
    return dir_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Circos input generation for all TFs')
    parser.add_argument('--dataset_path', default="data_files/all_results", help='name of folder having data files')
    parser.add_argument('--window_len', type=int, default=5, help='specify the length of sliding window')
    args, unknown = parser.parse_known_args()
    
    
    dataset_path = args.dataset_path
    #extract testing output filenames
    test_files = extract_files(dataset_path)
    
    #extract top 10 enriched motifs (based on adjusted p-value) from each TF
    test_df = extract_res(test_files,dataset_path,10)
    
    #writes circos input files
    dir_path = circos_all(test_df,dataset_path,args,10)
    
    #writes dataframe to text file
    text_file = dir_path + "/" + "top_enriched_motifs.txt"
    test_df.to_csv(text_file, sep="\t",index=False)
