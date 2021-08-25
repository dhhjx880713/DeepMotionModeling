import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../..'))
import numpy as np
import tensorflow as tf
from mosi_dev_deepmotionmodeling.utilities.utils import export_point_cloud_data_without_foot_contact
from mosi_dev_deepmotionmodeling.models.motion_vae_encoder import MotionVaeEncoder
tf.compat.v1.disable_eager_execution()


def demo_vae():
    ### load model
    data_path = os.path.join(dirname, '../..', 'data/training_data/dancing/encoded_data.npy')
    training_data = np.load(data_path)
    model_name = 'dance_vae'
    model_file = os.path.join(dirname, '../..', 'data/models', model_name, model_name + '.ckpt')
    npc = 32    
    input_dims = training_data.shape[1]
    N = 100  ### number of samples
    vae_model = MotionVaeEncoder(npc=npc, input_dims=input_dims, name=model_name,
                                 encoder_activation=tf.nn.elu,
                                 decoder_activation=tf.nn.elu,
                                 n_random_samples=1)
    vae_model.create_model()
    vae_model.load_model(model_file)
    new_samples = vae_model.generate_new_samples(N)
    print(new_samples.shape)
    np.save(os.path.join(os.path.dirname(data_path), 'sampled_motions.npy'), new_samples)
    ### decode motions
    ########## load dancing autoencoder to decoder sample mtoion to motion space
    ########## load training data for meta data

    training_data_path = os.path.join(dirname, '../..', r'data/training_data/dancing/dancing_data.npy')
    training_data = np.load(training_data_path)
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
    output_dim = training_data.shape[1]
    dropout_rate = 0.1

    mean_value = training_data.mean(axis=0)[np.newaxis, :]
    std_value = training_data.std(axis=0)[np.newaxis, :]
    std_value[std_value<EPS] = EPS  
    model_name = os.path.join(dirname, '../..', 'data\models\dancing_ClipEnc\dancing_ClipEnc_0150.ckpt')
    decoder = FullBodyPoseEncoder(output_dim=output_dim, dropout_rate=dropout_rate, npc=256)
    decoder.load_weights(model_name)    
 
    decoded_motions = decoder.decode(new_samples)

    decoded_motions = decoded_motions * std_value + mean_value
    decoded_motions = decoded_motions.eval(session=tf.compat.v1.Session())

    decoded_motions = np.reshape(decoded_motions, (N, 120, 90))
    print(type(decoded_motions))

    save_path = os.path.join(dirname, '../..', 'data/tmp')
    for i in range(len(decoded_motions)):
        save_filename = os.path.join(save_path, 'dancing_' + str(i) + '.panim')
        export_point_cloud_data_without_foot_contact(decoded_motions[i], save_filename, scale_factor=5)



if __name__ == "__main__":
    demo_vae()