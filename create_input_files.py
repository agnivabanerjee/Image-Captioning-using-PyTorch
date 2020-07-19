from utils import create_input_files
import _config

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset=_config.dataset,
                       karpathy_json_path=_config.karpathy_json_path,
                       image_folder=_config.image_foldr_path,
                       captions_per_image=_config.captions_per_image,
                       min_word_freq=_config.min_word_freq,
                       output_folder=_config.output_folder,
                       max_len=_config.max_sentence_length)
