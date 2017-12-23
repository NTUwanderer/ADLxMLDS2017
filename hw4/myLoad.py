from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

MODEL_PATH="skip_thoughts_uni_2017_02_02/"
VOCAB_FILE = MODEL_PATH + "vocab.txt"
EMBEDDING_MATRIX_FILE = MODEL_PATH + "embeddings.npy"
CHECKPOINT_PATH = MODEL_PATH + "model.ckpt-501424"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                    vocabulary_file=VOCAB_FILE,
                    embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                    checkpoint_path=CHECKPOINT_PATH)

result = encoder.encode(['brown eyes', 'white hair'])
print ('result: ', result)
