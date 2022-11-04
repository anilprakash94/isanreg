import argparse
import re
import random


def all_motifseq(args):
    with open(args.target_path, 'r') as file_lines:
        meme_line = file_lines.readlines()
    nucs = "A", "C", "G", "T"
    motif_seq = {}
    for line in range(len(meme_line)):
        if (meme_line[line].startswith('MOTIF') and meme_line[line+2].startswith('letter')):
            motif_name = meme_line[line].strip().replace('(','').replace(')',' ').replace('_','').split(" ")[2]
            motif_len = int(re.search("w= (.+?) ",meme_line[line+2]).group(1))
            for i in range(line+3,line+3+motif_len):
                freq_list = ([float(x) for x in meme_line[i].split()])
                max_index = freq_list.index(max(freq_list))
                nuc_letter = nucs[max_index]
                if motif_name not in motif_seq:
                    motif_seq[motif_name] = nuc_letter 
                motif_seq[motif_name] += nuc_letter
    return motif_seq


def generate_input(length):
    nucs = "ACGT"
    freq = [0.27, 0.23, 0.23, 0.27]
    freq = [x * length for x in freq]
    freq_dict = {}
    freq_dict["A"], freq_dict["C"], freq_dict["G"], freq_dict["T"] = freq
    nuc_num = {}
    nuc_num["A"], nuc_num["C"], nuc_num["G"], nuc_num["T"] = 0, 0, 0, 0
    input_seq = ''
    for i in range(length):
        random_nuc = random.choice(nucs)
        nuc_num[random_nuc] += 1
        input_seq += random_nuc
        if nuc_num[random_nuc] == freq_dict[random_nuc]:
            nucs = nucs.replace(random_nuc, '')
        else:
            pass
    return input_seq


def input_emded(train_firstidx, train_secondidx, args, first_tf, second_tf, embed="first"):
    if embed == "first":
        input_seq = generate_input(args.seq_len)
        motif1_start = random.choice(train_firstidx)
        input_seq = input_seq[:motif1_start] + first_tf + input_seq[motif1_start+len(first_tf):]
        x_test = input_seq
        y_test = 0
    elif embed == "second":
        input_seq = generate_input(args.seq_len)
        motif2_start = random.choice(train_secondidx)
        input_seq = input_seq[:motif2_start] + second_tf + input_seq[motif2_start+len(second_tf):]
        x_test = input_seq
        y_test = 0   
    else:
        input_seq = generate_input(args.seq_len)
        motif1_start = random.choice(train_firstidx)
        input_seq = input_seq[:motif1_start] + first_tf + input_seq[motif1_start+len(first_tf):]
        motif2_start = random.choice(train_secondidx)
        input_seq = input_seq[:motif2_start] + second_tf + input_seq[motif2_start+len(second_tf):]
        x_test = input_seq
        y_test = 1
    return x_test, y_test

def input_file(train_firstidx, train_secondidx, args, first_tf, second_tf, file_size, file_name):
    indices = list(range(file_size//3))
    with open(file_name, 'w') as output:
        for idx in indices:
            x_test, y_test = input_emded(train_firstidx, train_secondidx, args, first_tf, second_tf, embed="first")
            output.write(x_test + "\t" + str(y_test) + "\n")
            x_test, y_test = input_emded(train_firstidx, train_secondidx, args, first_tf, second_tf, embed="second")
            output.write(x_test + "\t" + str(y_test) + "\n")
            x_test, y_test = input_emded(train_firstidx, train_secondidx, args, first_tf, second_tf, embed="both")
            output.write(x_test + "\t" + str(y_test) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulated input generation')
    parser.add_argument('--data_size', type=int, default=180000, help='training size') 
    parser.add_argument('-t1', '--tf_1', default="CTCF", help='First TF name')
    parser.add_argument('-t2', '--tf_2', default="E2F1", help='Second TF name')
    parser.add_argument('--seq_len', type=int, default=2000, help='total sequence length for each input') 
    parser.add_argument('--target_path', default = "Homo_sapiens.meme", help='specify the target motif file path used by Tomtom for motif comparison')
    parser.add_argument('--train_file', default = "data_files/simulated_train2k.txt", help='specify the output filename for saving input data for training')
    parser.add_argument('--val_file', default = "data_files/simulated_val2k.txt", help='specify the output filename for saving input data for validation')
    parser.add_argument('--test_file', default = "data_files/simulated_test2k.txt", help='specify the output filename for saving input data for testing')
    args, unknown = parser.parse_known_args()
     
    
    print("Extracting highest affinity sequence of all TFs...")
    motif_seq = all_motifseq(args)
    
    first_tf = motif_seq[args.tf_1]
    second_tf = motif_seq[args.tf_2]
    
    left_nuc = (args.seq_len//2) - 100
    right_nuc = (args.seq_len//2) + 100
    #both TF motifs will be >15bp apart 
    left_idx = list(range(0, left_nuc - (len(first_tf) + 15)))
    right_idx = list(range(right_nuc+15, args.seq_len-len(first_tf)))
    first_indices = left_idx + right_idx
    second_indices = list(range(left_nuc, right_nuc-len(second_tf)))
    
    
    print("Writing simulated data to file for training...")
    input_file(first_indices, second_indices, args, first_tf, second_tf, args.data_size, args.train_file)
    
    val_size = (args.data_size * 5) // 100
    print("Writing simulated data to file for validation...")
    input_file(first_indices, second_indices, args, first_tf, second_tf, val_size, args.val_file)
    
    test_size = (args.data_size * 5) // 100
    print("Writing simulated data to file for testing...")
    input_file(first_indices, second_indices, args, first_tf, second_tf, test_size, args.test_file)
    
