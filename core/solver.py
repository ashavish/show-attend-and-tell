import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from utils import *
from core.utils import *
from bleu import evaluate
from core.bleu import evaluate
import matplotlib
import hickle

# matplotlib.use('Agg')

def load_features(chunk,split="train"):
    fname = split + ".features_"+str(chunk) + ".pkl"
    data_path = './data/' + split
    print("loading {}".format(fname))
    features = load_pickle(os.path.join(data_path, fname))
    return features

class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.chunk_size = kwargs.pop('chunk_size',10000)
        
        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer   

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):
        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        chunk_size = self.chunk_size
        # image_chunks = self.data['image_chunks']
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        # features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        filename_seq = self.data['annotations']['file_name']
        val_filename_seq = self.val_data['annotations']['file_name']
        # val_features = self.val_data['features']
        n_val_features = len(self.val_data['file_names'])
        n_iters_val = int(np.ceil(float(n_val_features)/self.batch_size))
        n_iters_val_chunk = int(np.ceil(float(chunk_size)/self.batch_size))

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=20)

       # tf.get_variable_scope().reuse_variables()
       # _, _, generated_captions = self.model.build_sampler(max_len=20)

        # train op
        #with tf.name_scope('optimizer'):
        with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
           
        # summary op   
        # tf.scalar_summary('batch_loss', loss)
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            #tf.histogram_summary(var.op.name, var)
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            #tf.histogram_summary(var.op.name+'/gradient', grad)
            tf.summary.histogram(var.op.name+'/gradient', grad)

        #summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch
        
        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)
            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)
            prev_loss = -1
            curr_loss = 0
            start_t = time.time()
            chunk_iter = int(np.ceil(float(n_examples)/chunk_size))
            chunk_start = [chunk_size*i for i in range(0,chunk_iter)]
            chunk_end = [chunk_size*i for i in range(1,chunk_iter+1)]
            chunk_end[-1] = n_examples
            for e in range(self.n_epochs):
                track_iter = 0
                for chnk in range(chunk_iter):
                    captions_chunk = captions[chunk_start[chnk]:chunk_end[chnk]]
                    image_idxs_chunk = image_idxs[chunk_start[chnk]:chunk_end[chnk]]
                    chnk_size = chunk_end[chnk] - chunk_start[chnk]
                    filename_chunk = np.array(filename_seq[chunk_start[chnk]:chunk_end[chnk]].tolist())
                    # Load feature chunk
                    image_feats = load_features(chnk,"train")
                    features_chunk = np.array([image_feats[each] for each in filename_chunk])
                    rand_idxs = np.random.permutation(chnk_size)
                    captions_ = captions_chunk[rand_idxs]
                    image_idxs_ = image_idxs_chunk[rand_idxs]
                    features_chunk = features_chunk[rand_idxs]
                    filename_chunk = filename_chunk[rand_idxs]                    
                    n_iters_per_chunk = int(np.ceil(float(chnk_size)/self.batch_size))
                    for i in range(n_iters_per_chunk):
                        captions_batch = captions_[i*self.batch_size:(i+1)*self.batch_size]
                        image_idxs_batch = image_idxs_[i*self.batch_size:(i+1)*self.batch_size]
                        features_batch = features_chunk[i*self.batch_size:(i+1)*self.batch_size]
                        filename_batch = filename_chunk[i*self.batch_size:(i+1)*self.batch_size]
                        # features_batch = get_features(image_idxs_batch)
                        feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                        _, l = sess.run([train_op, loss], feed_dict)
                        curr_loss += l
                        track_iter = track_iter + i
                        # write summary for tensorboard visualization
                        if track_iter % 10 == 0:
                            summary = sess.run(summary_op, feed_dict)
                            summary_writer.add_summary(summary, e*n_iters_per_epoch + track_iter)
                        if (track_iter+1) % self.print_every == 0:
                            print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, track_iter+1, l)
                            ground_truths = captions[image_idxs == image_idxs_batch[0]]
                            decoded = decode_captions(ground_truths, self.model.idx_to_word)
                            for j, gt in enumerate(decoded):
                                print "Ground truth %d: %s" %(j+1, gt)                    
                            gen_caps = sess.run(generated_captions, feed_dict)
                            decoded = decode_captions(gen_caps, self.model.idx_to_word)
                            print "Generated caption: %s\n" %decoded[0]
                    print("chunk completed ",str(chnk))                            
                    features_chunk = []
                    features_batch = []
                    feed_dict = {}    
                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0 
                # print out BLEU scores and file write
                if self.print_bleu:
                    val_chunk_iter = int(np.ceil(float(n_val_features)/chunk_size))
                    val_chunk_start = [chunk_size*i for i in range(0,val_chunk_iter)]
                    val_chunk_end = [chunk_size*i for i in range(1,val_chunk_iter+1)]
                    val_chunk_end[-1] = n_val_features
                    all_gen_cap = np.ndarray([])
                    # Insert chunking for loop
                    # load val features in chunk
                    for val_chnk in range(val_chunk_iter):
                        val_features = load_features(val_chnk,"val")
                        #all_gen_cap_chunk = np.ndarray((len(val_features.keys()), 20)) 
                        val_chunk_size =  val_chunk_end[val_chnk] -  val_chunk_start[val_chnk]   
                        n_iters_val_chunk = int(np.ceil(float(val_chunk_size)/self.batch_size))
                        val_filename_chunk = np.array(val_filename_seq[val_chunk_start[val_chnk]:val_chunk_end[val_chnk]].tolist())
                        val_features_chunk = np.array([val_features[each] for each in val_filename_chunk])
                        all_gen_cap = np.array([])
                        for i in range(n_iters_val_chunk):
                            val_indx = range(i*self.batch_size,(i+1)*self.batch_size)
                            if (i+1)*self.batch_size > val_chunk_size:
                                val_indx = range(i*self.batch_size,val_chunk_size)
                            features_batch = val_features_chunk[val_indx]
                            feed_dict = {self.model.features: features_batch}
                            gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  
                            #all_gen_cap_chunk[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                            if all_gen_cap.shape[0] ==0:
                                all_gen_cap = gen_cap
                            else:
                                all_gen_cap = np.concatenate((all_gen_cap,gen_cap),axis=0)    
                            # all_gen_cap = np.append(all_gen_cap,gen_cap)
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)
                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)

                    
         
    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        # features = data['features']
        chnk = 0
        image_feats = load_features(chnk,split)
        files = data['annotations']['file_name']
        chunk_size = self.chunk_size
        features = np.array([image_feats[each] for each in files[:chunk_size]])
        data_chunk = data
        data_chunk['file_names'] = files[:chunk_size]
        data_chunk['features'] = features
        # build a graph to sample captions
	    #alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            tf.get_variable_scope().reuse == False
            alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(data_chunk, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            image_files = image_files.tolist()
  
            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" %decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights 
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        fname = image_files[n].split("/")[-1].split(".")[0] + "_" + str(t) + ".jpeg"
                        print(fname)
                        plt.savefig(fname, dpi=120)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)  
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))
