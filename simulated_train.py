import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import os
import numpy as np
import math
import argparse
import pickle
import re
import random


def extract_peaks(train_file):
    input_seq = train_file
    train_peaks = []
    with open(input_seq, 'r') as f:
        file = f.readlines()
    
    for i in file:
        single_peak = i.strip().split("\t")
        train_peaks.append(single_peak)
    return train_peaks


def vectorization(seq_data):
    input_array = [] 
    nuc = 'NAGCT'
    encode = dict((allele, index) for index, allele in enumerate(nuc))
    for sample in seq_data:
        pos = []
        for nuc_base in sample:
            pos.append(encode[nuc_base.upper()])
        input_array.append(pos)
    return input_array


def create_mask(input_data, dim):
    mask = (input_data == 0)
    mask_array = tf.cast(mask, tf.float32)
    mask_array = tf.tile(tf.expand_dims(mask_array, axis=-1), [1,1,dim])
    mask_array = tf.constant([1], dtype= tf.float32) - mask_array
    return mask_array


def pos_encode(seq_len, dim):
    pos = np.tile(np.expand_dims(np.arange(seq_len, dtype=np.float32), 1), (1,dim))
    i = np.arange(dim, dtype=np.float32)
    for x,y in enumerate(pos):
        pos[x] = y / (10000 ** ((2 * (i//2)) / dim ) )
    pos[:, 0::2] = np.sin(pos[:, 0::2])
    pos[:, 1::2] = np.cos(pos[:, 1::2])    
    return pos


def pos_encode_batch(input_emb, mask_array, pos):
    pos = tf.tile(tf.expand_dims(pos, axis=0), [K.eval(K.shape(input_emb)[0]),1,1])
    pos = pos * mask_array
    input_pos = input_emb + pos
    return input_pos


def convert_to_blocks(input_pos, block_num, seq_len, dim):
    block_len = seq_len // block_num
    input_blk = tf.reshape(input_pos, (K.eval(K.shape(input_pos)[0]),block_num,block_len,dim) ) 
    return input_blk


class Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self, train_peaks, args):
        self.train_peaks = train_peaks
        self.batch_size = args.batch_size
        self.indices = np.arange(len(self.train_peaks))
        self.on_epoch_end()
        self.iterate = 0
    
    def __len__(self):
        return len(self.train_peaks) // self.batch_size
    
    def __getitem__(self, idx):
        index_list = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_train = [self.train_peaks[index] for index in index_list]
        output_array = [int(i[1]) for i in batch_train]
        y_train = tf.stack(output_array, axis=0)
        y_train = tf.expand_dims(y_train, axis=-1)
        seq_data = [i[0] for i in batch_train]
        input_array = vectorization(seq_data)
        x_train = tf.stack(input_array)
        return x_train, y_train
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __next__(self):
        if self.iterate >= self.__len__():
            self.iterate = 0
        self.iterate += 1
        return self.__getitem__(self.iterate-1)


def weights_filepath(model_dir):
    current_dir = os.getcwd()
    dir_path = os.path.join(current_dir,model_dir)
    if os.path.isdir(dir_path):
        pass   
    else:
        os.makedirs(dir_path)
    return dir_path


class emb2block(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def call(self, inp_tensors):
        dim = self.args.dim
        seq_len = self.args.seq_len
        block_num = self.args.block_num
        input_emb = inp_tensors[0]
        input_data = inp_tensors[1]
        mask_array = create_mask(input_data, dim)
        input_emb = input_emb * mask_array
        pos = pos_encode(seq_len, dim)
        input_pos = pos_encode_batch(input_emb, mask_array, pos)
        input_blk = convert_to_blocks(input_pos, block_num, seq_len, dim)
        return [input_blk, input_pos, mask_array]

class MultiHead_attn(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.w1 = self.add_weight(name='w1', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.w2 = self.add_weight(name='w2', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.w3 = self.add_weight(name='w3', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.b1 = self.add_weight(name='b1', shape=(self.dim,), initializer="zeros", trainable=True)
        self.b2 = self.add_weight(name='b2', shape=(self.dim,), initializer="zeros", trainable=True)
        self.b3 = self.add_weight(name='b3', shape=(self.dim,), initializer="zeros", trainable=True)
    
    def call(self, inp_tensors, masking = True, training= None):
        dim = self.args.dim
        heads = self.args.heads
        block_num = self.args.block_num
        seq_len = self.args.seq_len
        block_len = seq_len // block_num
        inputs, mask_array = inp_tensors
        Query = tf.matmul(inputs, self.w1) + self.b1
        Key = tf.matmul(inputs, self.w2) + self.b2
        Value = tf.matmul(inputs, self.w3) + self.b3
        Query = tf.concat(tf.split(Query, heads, axis=-1), axis=0)
        Key = tf.concat(tf.split(Key, heads, axis=-1), axis=0)
        Value = tf.concat(tf.split(Value, heads, axis=-1), axis=0)   
        if K.shape(inputs).shape[0] == 4:
            res = (tf.matmul(Query, tf.transpose(Key, [0, 1, 3, 2]))) / math.sqrt(K.int_shape(Key)[-1]) 
            if masking:
                mask_array = tf.reshape(mask_array, (K.int_shape(mask_array)[0],block_num,block_len,dim) )
                mask_array = tf.concat(tf.split(mask_array, heads, axis=3), axis=0)
                if block_len > K.int_shape(mask_array)[-1]:
                    mask_array= tf.tile(tf.slice(mask_array, [0,0,0,0], [K.int_shape(mask_array)[0],block_num,block_len,1]), [1,1,1,block_len])    
                elif block_len < K.int_shape(mask_array)[-1]:
                    mask_array = tf.slice(mask_array, [0,0,0,0], [K.int_shape(mask_array)[0],block_num,block_len,block_len])
                mask_array = tf.constant([1], dtype= tf.float32) - mask_array
                mask_array = mask_array * tf.constant([-1e10], dtype= tf.float32)
                res = res + mask_array
            res = tf.nn.softmax(res)
            if training:
                res = tf.nn.experimental.stateless_dropout(res, rate=0.2, seed=[1, 0])
            res = tf.matmul(res, Value)
            res = tf.concat(tf.split(res, heads, axis=0), axis=-1)
            res = res + inputs
            res = tf.keras.layers.LayerNormalization(axis=-1)(res)
            return res
        
        elif K.shape(inputs).shape[0] == 3:
            out = (tf.matmul(Query, tf.transpose(Key, [0, 2, 1]))) / math.sqrt(K.int_shape(Key)[-1])
            if masking:
                mask_array = tf.reshape(mask_array, (K.int_shape(mask_array)[0],block_num,block_len,dim) )
                mask_array = tf.concat(tf.split(mask_array, heads, axis=3), axis=0)
                mask_array = tf.math.reduce_max(mask_array, axis= -2)
                if block_num > K.int_shape(mask_array)[-1]:
                    mask_array= tf.tile(tf.slice(mask_array, [0,0,0], [K.int_shape(mask_array)[0],block_num,1]), [1,1,block_num])
                elif block_num < K.int_shape(mask_array)[-1]:
                    mask_array = tf.slice(mask_array, [0,0,0], [K.int_shape(mask_array)[0],block_num,block_num])
                mask_array = tf.constant([1], dtype= tf.float32) - mask_array
                mask_array = mask_array * tf.constant([-1e10], dtype= tf.float32)
                out = out + mask_array
            out = tf.nn.softmax(out)
            if training:
                out = tf.nn.experimental.stateless_dropout(out, rate=0.2, seed=[1, 0])
            out = tf.matmul(out, Value)
            out = tf.concat(tf.split(out, heads, axis=0), axis=-1)
            out = out + inputs
            out = tf.keras.layers.LayerNormalization(axis=-1)(out)
            return out


class Dotproduct_scaled(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.dw1 = self.add_weight(name='dw1', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.dw2 = self.add_weight(name='dw2', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.dw3 = self.add_weight(name='dw3', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.db1 = self.add_weight(name='db1', shape=(self.dim,), initializer="zeros", trainable=True)
        self.db2 = self.add_weight(name='db2', shape=(self.dim,), initializer="zeros", trainable=True)
        self.db3 = self.add_weight(name='db3', shape=(self.dim,), initializer="zeros", trainable=True)
    
    def call(self, inp_tensors, masking = True, training= None):
        dim = self.args.dim
        block_num = self.args.block_num
        seq_len = self.args.seq_len
        block_len = seq_len // block_num
        inputs, mask_array = inp_tensors
        Query = tf.matmul(inputs, self.dw1) + self.db1
        Key = tf.matmul(inputs, self.dw2) + self.db2
        Value = tf.matmul(inputs, self.dw3) + self.db3
        if K.shape(inputs).shape[0] == 4:
            res = (tf.matmul(Query, tf.transpose(Key, [0, 1, 3, 2]))) / math.sqrt(K.int_shape(Key)[-1])
            if masking:
                mask_array = tf.reshape(mask_array, (K.int_shape(mask_array)[0],block_num,block_len,dim) ) 
                if block_len > dim:
                    mask_array= tf.tile(tf.slice(mask_array, [0,0,0,0], [K.int_shape(mask_array)[0],block_num,block_len,1]), [1,1,1,block_len])
                elif block_len < dim:
                    mask_array = tf.slice(mask_array, [0,0,0,0], [K.int_shape(mask_array)[0],block_num,block_len,block_len])
                mask_array = tf.constant([1], dtype= tf.float32) - mask_array
                mask_array = mask_array * tf.constant([-1e10], dtype= tf.float32)
                res = res + mask_array
            res = tf.nn.softmax(res)
            if training:
                res = tf.nn.experimental.stateless_dropout(res, rate=0.2, seed=[1, 0])
            res = tf.matmul(res, Value)
            res = res + inputs
            res = tf.keras.layers.LayerNormalization(axis=-1)(res)
            return res
        
        elif K.shape(inputs).shape[0] == 3:
            out = (tf.matmul(Query, tf.transpose(Key, [0, 2, 1]))) / math.sqrt(K.int_shape(Key)[-1])
            if masking:
                mask_array = tf.reshape(mask_array, (K.int_shape(mask_array)[0],block_num,block_len,dim) )        
                mask_array = tf.math.reduce_max(mask_array, axis= -2)
                if block_num > dim:
                    mask_array= tf.tile(tf.slice(mask_array, [0,0,0], [K.int_shape(mask_array)[0],block_num,1]), [1,1,block_num])
                elif block_num < dim:
                    mask_array = tf.slice(mask_array, [0,0,0], [K.int_shape(mask_array)[0],block_num,block_num])
                mask_array = tf.constant([1], dtype= tf.float32) - mask_array
                mask_array = mask_array * tf.constant([-1e10], dtype= tf.float32)
                out = out + mask_array
            out = tf.nn.softmax(out)
            if training:
                out = tf.nn.experimental.stateless_dropout(out, rate=0.2, seed=[1, 0])
            out = tf.matmul(out, Value)
            out = out + inputs
            out = tf.keras.layers.LayerNormalization(axis=-1)(out)
            return out


class srct_att(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.w4 = self.add_weight(name='w4', shape=(self.dim, self.dim), initializer="he_uniform", trainable=True)
        self.w5 = self.add_weight(name='w5', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.b4 = self.add_weight(name='b4', shape=(self.dim,), initializer="zeros", trainable=True)
        self.b5 = self.add_weight(name='b5', shape=(self.dim,), initializer="zeros", trainable=True)
    
    def call(self, res, final_context = None, att_weight = None):
        dim = self.args.dim
        seq_len = self.args.seq_len
        srct = tf.nn.leaky_relu(tf.matmul(res, self.w4) + self.b4)
        srct = tf.matmul(srct, self.w5) + self.b5
        srct_att = tf.nn.softmax(srct, axis=-2)
        if final_context:
            if tf.is_tensor(att_weight):
                srct_att = att_weight  
            srct = srct_att * res
            srct = tf.keras.layers.LayerNormalization(axis=-1)(srct)
            return [srct, srct_att]
        else:
            srct = tf.math.reduce_sum(srct_att * res, axis=-2)
            srct = tf.keras.layers.LayerNormalization(axis=-1)(srct)
            return srct


class refined_gating(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.rw1 = self.add_weight(name='rw1', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.rw2 = self.add_weight(name='rw2', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        bias = np.random.uniform(low = 1/self.dim, high = 1-(1/self.dim), size = self.dim )
        bias = -np.log((1/bias)-1)
        self.rb1 = self.add_weight(name='rb1', shape=(self.dim,), initializer= tf.constant_initializer(bias), trainable=True)
    
    def call(self, out, srct):
        dim = self.args.dim
        block_num = self.args.block_num
        seq_len = self.args.seq_len
        block_len = seq_len // block_num
        forget = refined = tf.matmul(out, self.rw1) + tf.matmul(srct, self.rw2)
        forget_gate = tf.math.sigmoid(forget + self.rb1)
        refined_gate = tf.math.sigmoid(refined - self.rb1)
        effective_gate = refined_gate * (1-(1-forget_gate)**2) + (1-refined_gate) * forget_gate**2 
        input_gate = tf.constant([1], dtype= tf.float32) - effective_gate
        gate_output = (effective_gate * out) + (input_gate * srct )
        lc = tf.reshape(tf.tile(tf.expand_dims(gate_output, -2), [1, 1, block_len, 1]), [K.int_shape(gate_output)[0], seq_len, dim] )
        return lc


class refined_fusion_gate(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.fw1 = self.add_weight(name='fw1', shape=(self.dim * 3, self.dim), initializer="he_uniform", trainable=True)
        self.fw2 = self.add_weight(name='fw2', shape=(self.dim * 3, self.dim), initializer="glorot_uniform", trainable=True)
        self.fb1 = self.add_weight(name='fb1', shape=(self.dim,), initializer="zeros", trainable=True)
        bias = np.random.uniform(low = 1/self.dim, high = 1-(1/self.dim), size = self.dim )
        bias = -np.log((1/bias)-1)
        self.fb2 = self.add_weight(name='fb2', shape=(self.dim,), initializer= tf.constant_initializer(bias), trainable=True)
    
    def call(self, input_pos, res, lc):
        dim = self.args.dim
        seq_len = self.args.seq_len
        res = tf.reshape(res, [K.int_shape(res)[0], seq_len, dim])
        fusion =  tf.matmul(tf.concat([input_pos, res, lc], axis=-1), self.fw1)
        fusion = tf.nn.leaky_relu(fusion + self.fb1)
        forget = refined = tf.matmul(tf.concat([input_pos, res, lc], axis=-1), self.fw2)
        forget_gate = tf.math.sigmoid(forget + self.fb2)
        refined_gate = tf.math.sigmoid(refined - self.fb2) 
        effective_gate = refined_gate * (1-(1-forget_gate)**2) + (1-refined_gate) * forget_gate**2
        input_gate = tf.constant([1], dtype= tf.float32) - effective_gate
        output = (effective_gate * fusion) + (input_gate * input_pos)
        output = tf.keras.layers.LayerNormalization(axis=-1)(output)
        return output


class squeeze_tensor(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.sw1 =  self.add_weight(name='sw1', shape=(self.dim, 1), initializer="he_uniform", trainable=True)
        self.sb1 = self.add_weight(name='sb1', shape=(1,), initializer="zeros", trainable=True)
    
    def call(self, output):
        output = tf.matmul(output, self.sw1)
        output = tf.nn.leaky_relu(output + self.sb1)
        output = tf.squeeze(output, axis= -1)  
        output = tf.keras.layers.LayerNormalization(axis=-1)(output)
        return output

class opt_weights(tf.keras.callbacks.Callback):
    def __init__(self, dir_path, opt): 
        super().__init__()
        self.dir_path = dir_path
        self.opt = opt
    
    def on_epoch_end(self, epoch, logs={}):
        opt_weights = tf.keras.optimizers.Adam.get_weights(self.opt)
        with open(self.dir_path+'/optimizer.pkl', 'wb') as f:
            pickle.dump(opt_weights, f)



class Modelsubclass(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.embed1 = tf.keras.layers.Embedding(5, args.dim)
        self.block = emb2block(args)
        self.dense1 = tf.keras.layers.Dense(args.dim, activation=None, kernel_initializer="he_uniform")
        self.relu1 = tf.keras.layers.LeakyReLU()
        self.drop1 = tf.keras.layers.Dropout(.2)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.dense2 = tf.keras.layers.Dense(args.dim, activation=None)
        self.drop2 = tf.keras.layers.Dropout(.2)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        if args.attention == "multi_head":
            self.att1 = MultiHead_attn(args)
            self.srct1 = srct_att(args)
            self.att2 = MultiHead_attn(args)
        elif args.attention == "dot_product":
            self.att1 = Dotproduct_scaled(args)
            self.srct1 = srct_att(args)
            self.att2 = Dotproduct_scaled(args)
        self.refine1 = refined_gating(args)
        self.fusion1 = refined_fusion_gate(args)
        self.srct2 = srct_att(args)
        self.squeeze = squeeze_tensor()
        self.dense3 = tf.keras.layers.Dense(1, activation=None)
    
    def call(self, input_data, masking=True, training=False, attr= None, att_weight = None):
        input_emb = self.embed1(input_data)
        block_output =  self.block([input_emb, input_data])
        inputs = self.dense1(block_output[0])
        inputs = self.relu1(inputs)
        inputs = self.drop1(inputs)
        inputs = self.norm1(inputs)
        inputs = self.dense2(inputs)
        inputs = self.drop2(inputs)
        inputs = self.norm2(inputs)
        res = self.att1([inputs, block_output[2]], training = training)
        srct = self.srct1(res, final_context = None)
        out = self.att2([srct, block_output[2]], training = training)
        lc = self.refine1(out, srct)
        output = self.fusion1(block_output[1], res, lc)
        srct2_output = self.srct2(output, final_context = True, att_weight = att_weight)
        output = self.squeeze(srct2_output[0])
        if attr:
            output = tf.math.sigmoid(self.dense3(output))
            return output, srct2_output[1]
        else:
            output = tf.math.sigmoid(self.dense3(output))
            return output


def one_train_batch(args, train_peaks):
    batch_train = train_peaks[0:1]
    output_array = [int(i[1]) for i in batch_train]
    y_train = tf.stack(output_array, axis=0)
    seq_data = [i[0] for i in batch_train]
    input_array = vectorization(seq_data)
    x_train = tf.stack(input_array)
    return x_train, y_train

@tf.autograph.experimental.do_not_convert
def custom_loss(y_train, y_pred):
    y_train = tf.cast(y_train, tf.float32)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    pos = 2 * y_train * K.log(y_pred)
    neg = (1 - y_train) * K.log(1 - y_pred)
    loss = -K.mean(pos + neg, axis = 1)
    return loss


def model_run(args, train_peaks, val_peaks):
    print("Instantiating model...")
    model = Modelsubclass(args)
    opt=tf.keras.optimizers.Adam(learning_rate=args.lr)
    dir_path = weights_filepath(args.model_dir+"/simulated")
    
    save_opt = opt_weights(dir_path, opt)
    
    acc = tf.keras.metrics.BinaryAccuracy()
    
    print("Compiling model...")
    model.compile(loss=custom_loss, optimizer=opt, metrics=[acc], run_eagerly=True) 
    
    train_generator = Custom_Generator(train_peaks, args)

    val_generator = Custom_Generator(val_peaks, args)
    
    resume_train = args.resume_train
    log_name = os.getcwd() + "/" + args.model_dir + "/" + args.log_file
        
    if resume_train == False:
        csv_logger = tf.keras.callbacks.CSVLogger(log_name)
    else:
        csv_logger = tf.keras.callbacks.CSVLogger(log_name, append=True)
    
    weights_path = os.path.join(dir_path, "weights.{epoch:02d}-{val_loss:.2f}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=False)

    
    if resume_train == False:
        print("Training new model...")
        model.fit(x=train_generator, epochs=500, validation_data=val_generator, callbacks=[model_checkpoint_callback, save_opt, csv_logger], verbose= 1)
    else:
        checkpoint_file = dir_path + "/checkpoint"
        with open(checkpoint_file, 'r') as f:
            file = f.readlines()
        
        weights_file = re.search('path: "(.+?)"', file[0]).group(1)
        epoch_num = int(re.search('weights.(.+?)-', weights_file).group(1))
        weights_file = os.path.join(dir_path, weights_file)
        
        print("Loading weights from previously trained model...")
        status = model.load_weights(weights_file).expect_partial()
        status.assert_existing_objects_matched()
        
        print("Training single input batch...")
        x_single, y_single = one_train_batch(args, train_peaks)
        model.fit(x_single, y_single, batch_size=args.batch_size)
        
        with open(dir_path+'/optimizer.pkl', 'rb') as f:
            weight_variable = pickle.load(f)
        
        print("Loading optimizer weights from file...")
        model.optimizer.set_weights(weight_variable)
        
        print("Resuming training of model from last saved epoch...")
        model.fit(x=train_generator, epochs=500, validation_data=val_generator, callbacks=[model_checkpoint_callback, save_opt, csv_logger], initial_epoch=epoch_num, verbose= 1)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulated Training')
    parser.add_argument('--dim', type=int, default=128, help='dimensions of input after embedding')
    parser.add_argument('--batch_size', type=int, default=20, help='specify the batch_size needed for training')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking+bin sequence length for each input') 
    parser.add_argument('--block_num', type=int, default=200, help='number of blocks into which the input sequence should be split, seq_len should be divisible by block_num')
    parser.add_argument('--heads', type=int, default=4, help='number of multi head attention heads')
    parser.add_argument('-att', '--attention', dest = "attention", default = "multi_head", help='specify the Intra-block and Inter-block self-attention method, "dot_product" can be used to reduce computation time')
    parser.add_argument('--resume_train', dest = "resume_train", action='store_true', help='default to False when the command-line argument is not present, if true already trained weights are loaded to resume training')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate float value')
    parser.add_argument('--model_dir', dest = "model_dir", default = "Models", help='specify the output folder name for saving the model')
    parser.add_argument('--log_file', default = "simulated_train.log", help='specify the name of training log file')
    parser.add_argument('--train_file', default = "data_files/simulated_train2k.txt", help='specify the input data for training')
    parser.add_argument('--val_file', default = "data_files/simulated_val2k.txt", help='specify the input data for validation')
    args, unknown = parser.parse_known_args()
    
    #raises exception if seq_len is not divisible by block_num
    
    if args.seq_len % args.block_num != 0:
        raise ValueError("seq_len is not divisible by block_num")   
    
    print("Extracting training data from file...")
    train_peaks = extract_peaks(args.train_file)
    
    print("Extracting validation data from file...")
    val_peaks = extract_peaks(args.val_file)
    
    
    #model run function
    model_run(args, train_peaks, val_peaks)
    
