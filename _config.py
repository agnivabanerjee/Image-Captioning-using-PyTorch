## the following parameters are intrinsic to create_input_files - used in train
dataset = 'coco'  # options are {'coco', 'flickr8k', 'flickr30k'}
karpathy_json_path = 'D:\\Datasets\\cv\\ms_coco\\caption_datasets\\dataset_coco.json'
image_foldr_path = 'D:\\Datasets\\cv\\ms_coco\\images\\'
captions_per_image = 5
min_word_freq = 5
output_folder = 'D:\\Datasets\\cv\\ICWithAttention\\output_folder\\'
max_sentence_length = 50

## the following parameters are intrinsic to train - used in eval
base_file_name = 'coco_5_cap_per_img_5_min_word_freq'
# Model parameters
word_embedding_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.2
# Training parameters
start_epoch = 0
epochs = 80  # number of epochs to train for (if early stopping is not triggered)
batch_size = 32
workers = 0  # for data-loading; right now, only 0 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint_path = None  # path to checkpoint, None if none
lr_shrink_factor = 0.8

## the following parameters are used in utils to preprocess images and text
image_channels = 3
image_dimension = 512
start_tag = '<start>'
pad_tag = '<pad>'
unk_tag = '<unk>'
end_tag = '<end>'

## the following parameters are used to evaluate performance
beam_search_size = 5
