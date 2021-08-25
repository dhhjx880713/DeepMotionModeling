import tensorflow as tf 
import numpy as np
POINT_LEN = 3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, ".."))
from tensorflow.keras import layers, Model
from collections import OrderedDict
from utilities.utils import get_global_position_framewise_tf
MSE = tf.keras.losses.MeanSquaredError()



def loss_fn(target, predict):
    return MSE(target, predict)

def loss_global_position(y_actual, y_pred):
    """measure MSE in global joint position space

    Args:
        y_actual : original motion
        y_pred : predicted motion
    """
    target_motion = get_global_position_framewise_tf(y_actual)
    predicted_motion = get_global_position_framewise_tf(y_pred)
    loss = MSE(target_motion, predicted_motion)
    return loss    


class FrameEncoderNoGlobal(tf.keras.Model):
    def __init__(self, dropout_rate, hidden_size=10, name=None):
        super(FrameEncoderNoGlobal, self).__init__(name=name)
        self.num_params_per_joint = 3
        self.z_unit = hidden_size
        self.animated_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
                                'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
                                'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
                                'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

        self.L1_groups = {'torso': ['Hips', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head'],
                          'leftArm': ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1'],
                          'rightArm': ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1'],
                          'leftLeg': ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'],
                          'rightLeg': ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']}
        self.L2_groups = {'torso_leftArm': ['torso', 'leftArm'],
                          'torso_rightArm': ['torso', 'rightArm'],
                          'torso_leftLeg': ['torso', 'leftLeg'],
                          'torso_rightLeg': ['torso', 'rightLeg']}
        self.L3_groups = {'upper_body': ['torso_leftArm', 'torso_rightArm'],
                          'lower_body': ['torso_leftLeg', 'torso_rightLeg']}
        self.L4_groups = {'full_body': ['upper_body', 'lower_body']}                                                

        # self.L1_units = 32
        # self.L2_units = 64
        # self.L3_units = 128
        # self.L4_units = 256
        self.L1_units = 64
        self.L2_units = 128
        self.L3_units = 256
        self.L4_units = 512
        self.activation = tf.nn.elu
        self.layers_dict = OrderedDict()
        self.encoder_net = OrderedDict()
        self.decoder_net = OrderedDict()

        ### L1 group ###
        for group in self.L1_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L1_units, activation=self.activation, name=group)                       

        ### L2 group ###
        for group in self.L2_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L2_units, activation=self.activation, name=group)
        
        ### L3 group ###
        for group in self.L3_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L3_units, activation=self.activation, name=group)

        ### L4 group ###
        for group in self.L4_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L4_units, activation=self.activation, name=group)    

        ### representation layer ###
        self.layers_dict['z_layer'] = layers.Dense(self.z_unit, activation=self.activation, name='z_layer')

        ### L4 inverse group ###
        for group in self.L4_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L4_units, activation=self.activation, name=group + '_inverse')            

        ### L3 inverse group ###
        for group in self.L3_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L3_units, activation=self.activation, name=group + '_inverse')

        ### L2 inverse group ###
        for group in self.L2_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L2_units, activation=self.activation, name=group + '_inverse')

        ### L1 inverse group ###
        for group in self.L1_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L1_units, activation=self.activation, name=group + '_inverse')

        ### output group ###
        for group in self.L1_groups.keys():

            self.layers_dict[group + '_output'] = layers.Dense(len(self.L1_groups[group]) * 3, name=group + '_output')   

        self.layers_dict['dropout'] = layers.Dropout(dropout_rate) 

    def _encoder(self, inputs, training=False):

        for group_name, values in self.L1_groups.items():
            ## preparing inputs

            self.encoder_net[group_name + '_input'] = tf.concat([inputs[:, self.animated_joints.index(joint)*self.num_params_per_joint:
                                                          (self.animated_joints.index(joint) + 1) * self.num_params_per_joint] for joint in values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'])
            # else:
            #     input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])                                    
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)


        ### encoder L2 ###
        for group_name, group_values in self.L2_groups.items():
            ### merge L1 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L3 ###

        for group_name, group_values in self.L3_groups.items():
            ### merge L2 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L4 ###

        for group_name, group_values in self.L4_groups.items():
            ### merge L3 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)

        ### create hidden representation
        # the input is fixed by the keyword 'full_body'
        # if training:
        #     self.encoder_net['full_body'] = self.layers_dict['dropout'](self.encoder_net['full_body'], training=training)
        self.encoder_net['z_code'] = self.layers_dict['z_layer'](self.encoder_net['full_body'])

        self.encoder_net['z_code'] = self.layers_dict['dropout'](self.encoder_net['z_code'], training=training)
        return self.encoder_net['z_code']

    def _decoder(self, inputs, training=False):
        ### decoder L4 ###
        # if training:
        #     inputs = self.layers_dict['dropout'](inputs, training=training)
        self.decoder_net['full_body'] = self.layers_dict['full_body_inverse'](inputs)  ## size 

        self.decoder_net['full_body'] = self.layers_dict['dropout'](self.decoder_net['full_body'], training=training)
        
        ### decoder L3 ###

        for group in self.L3_groups.keys():
            ### super group can be more than one
            super_groups = [group_name for group_name, values in self.L4_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            # if training:
            #     layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)

                
        ### decoder L2 ###
        for group in self.L2_groups.keys():

            super_groups = [group_name for group_name, values in self.L3_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            # if training:
            #     layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)

        ### decoder L1 ###
        for group in self.L1_groups.keys():

            super_groups = [group_name for group_name, values in self.L2_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            # if training:
            #     layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder output ###

        for group, joints in self.L1_groups.items():
            self.decoder_net[group + '_output'] = self.layers_dict[group + '_output'](self.decoder_net[group])

            for joint in joints:
                self.decoder_net[joint] = self.decoder_net[group + '_output'][:, joints.index(joint) * self.num_params_per_joint : (joints.index(joint) + 1) * self.num_params_per_joint]
        ### reorder 
        output = tf.concat([self.decoder_net[joint] for joint in self.animated_joints], axis=-1)
        return output
            
    def call(self, inputs, training=False):

        z_encoder = self._encoder(inputs, training=training)
        decoreded = self._decoder(z_encoder, training=training)
        return decoreded

class FrameEncoderDropoutFirst(tf.keras.Model):
    """spaial motion encoder based on skeleton hierarchy (only work for MakeHuman CMU skeleton)
    
    """
    def __init__(self, dropout_rate):
        super(FrameEncoderDropoutFirst, self).__init__()
        self.num_params_per_joint = 3
        self.z_unit = 10
        self.animated_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
                                'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
                                'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
                                'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

        self.L1_groups = {'torso': ['Hips', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head'],
                          'leftArm': ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1'],
                          'rightArm': ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1'],
                          'leftLeg': ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'],
                          'rightLeg': ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'],
                          'global_trans': ['velocity_x', 'velocity_z', 'velocity_r']}
        self.L2_groups = {'torso_leftArm': ['torso', 'leftArm'],
                          'torso_rightArm': ['torso', 'rightArm'],
                          'torso_leftLeg': ['torso', 'leftLeg'],
                          'torso_rightLeg': ['torso', 'rightLeg']}
        self.L3_groups = {'upper_body': ['torso_leftArm', 'torso_rightArm'],
                          'lower_body': ['torso_leftLeg', 'torso_rightLeg']}
        self.L4_groups = {'full_body': ['upper_body', 'lower_body', 'global_trans']}      
                                                  
        self.L1_units = 64
        self.L2_units = 128
        self.L3_units = 256
        self.L4_units = 512

        self.activation = tf.nn.elu
        self.layers_dict = OrderedDict()
        self.encoder_net = OrderedDict()
        self.decoder_net = OrderedDict()

        ### L1 group ###
        for group in self.L1_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L1_units, activation=self.activation, name=group)                       

        ### L2 group ###
        for group in self.L2_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L2_units, activation=self.activation, name=group)
        
        ### L3 group ###
        for group in self.L3_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L3_units, activation=self.activation, name=group)

        ### L4 group ###
        for group in self.L4_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L4_units, activation=self.activation, name=group)    

        ### representation layer ###
        self.layers_dict['z_layer'] = layers.Dense(self.z_unit, activation=self.activation, name='z_layer')

        ### L4 inverse group ###
        for group in self.L4_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L4_units, activation=self.activation, name=group + '_inverse')            

        ### L3 inverse group ###
        for group in self.L3_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L3_units, activation=self.activation, name=group + '_inverse')

        ### L2 inverse group ###
        for group in self.L2_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L2_units, activation=self.activation, name=group + '_inverse')

        ### L1 inverse group ###
        for group in self.L1_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L1_units, activation=self.activation, name=group + '_inverse')

        ### output group ###
        for group in self.L1_groups.keys():
            if group == "global_trans":
                self.layers_dict[group + '_output'] = layers.Dense(3, name=group + '_output')
            else:
                self.layers_dict[group + '_output'] = layers.Dense(len(self.L1_groups[group]) * 3, name=group + '_output')   

        self.layers_dict['dropout'] = layers.Dropout(dropout_rate) 

    def _encoder(self, inputs, training=False):

        for group_name, values in self.L1_groups.items():
            ## preparing inputs
            if group_name == "global_trans":
                self.encoder_net[group_name + '_input'] = inputs[:, -3:]  ## assume the last three dimensions are global transformation
            else:
                self.encoder_net[group_name + '_input'] = tf.concat([inputs[:, self.animated_joints.index(joint)*self.num_params_per_joint:
                                                          (self.animated_joints.index(joint) + 1) * self.num_params_per_joint] for joint in values], axis=-1)
            if training:
                input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            else:
                input = self.encoder_net[group_name + '_input']    
            self.encoder_net[group_name] = self.layers_dict[group_name](input)                               

        ### encoder L2 ###
        for group_name, group_values in self.L2_groups.items():
            ### merge L1 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            if training:
                input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            else:
                input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](input) 
          
        ### encoder L3 ###

        for group_name, group_values in self.L3_groups.items():
            ### merge L2 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            if training:
                input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            else:
                input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](input)           

        ### encoder L4 ###

        for group_name, group_values in self.L4_groups.items():
            ### merge L3 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            if training:
                input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            else:
                input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](input)

        ### create hidden representation
        # the input is fixed by the keyword 'full_body'
        if training:
            self.encoder_net['full_body'] = self.layers_dict['dropout'](self.encoder_net['full_body'], training=training)
        self.encoder_net['z_code'] = self.layers_dict['z_layer'](self.encoder_net['full_body'])

        return self.encoder_net['z_code']

    def _decoder(self, inputs, training=False):
        ### decoder L4 ###
        if training:
            inputs = self.layers_dict['dropout'](inputs, training=training)
        self.decoder_net['full_body'] = self.layers_dict['full_body_inverse'](inputs)
        
        ### decoder L3 ###
        for group in self.L3_groups.keys():
            ### super group can be more than one
            super_groups = [group_name for group_name, values in self.L4_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            if training:
                layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])
                
        ### decoder L2 ###
        for group in self.L2_groups.keys():

            super_groups = [group_name for group_name, values in self.L3_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            if training:
                layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder L1 ###
        for group in self.L1_groups.keys():

            if group == "global_trans":
                if training:
                    self.decoder_net['full_body'] = self.layers_dict['dropout'](self.decoder_net['full_body'], training=training)
                self.decoder_net[group] = self.layers_dict[group+'_inverse'](self.decoder_net['full_body'])
                # self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](self.decoder_net['full_body']), training=training)
            else:
                super_groups = [group_name for group_name, values in self.L2_groups.items() if group in values]
                layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
                if training:
                    layer_input = self.layers_dict['dropout'](layer_input, training=training)
                self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
                # self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder output ###

        for group, joints in self.L1_groups.items():
            self.decoder_net[group + '_output'] = self.layers_dict[group + '_output'](self.decoder_net[group])

            for joint in joints:
                self.decoder_net[joint] = self.decoder_net[group + '_output'][:, joints.index(joint) * self.num_params_per_joint : (joints.index(joint) + 1) * self.num_params_per_joint]
        ### reorder 
        output = tf.concat([self.decoder_net[joint] for joint in self.animated_joints], axis=-1)
        output = tf.concat([output, self.decoder_net['global_trans_output']], axis=-1)
        return output
            
    def call(self, inputs, training=False):
        z_encoder = self._encoder(inputs, training=training)
        decoreded = self._decoder(z_encoder, training=training)
        return decoreded




class FrameEncoder(tf.keras.Model):
    """spaial motion encoder based on skeleton hierarchy (only work for MakeHuman CMU skeleton)
    
    """
    def __init__(self, dropout_rate, hidden_size=10, name=None):
        super(FrameEncoder, self).__init__(name=name)
        self.num_params_per_joint = 3
        self.z_unit = hidden_size
        self.animated_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
                                'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
                                'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
                                'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

        self.L1_groups = {'torso': ['Hips', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head'],
                          'leftArm': ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase', 'LeftHandFinger1'],
                          'rightArm': ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1'],
                          'leftLeg': ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'],
                          'rightLeg': ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'],
                          'global_trans': ['velocity_x', 'velocity_z', 'velocity_r']}
        self.L2_groups = {'torso_leftArm': ['torso', 'leftArm'],
                          'torso_rightArm': ['torso', 'rightArm'],
                          'torso_leftLeg': ['torso', 'leftLeg'],
                          'torso_rightLeg': ['torso', 'rightLeg']}
        self.L3_groups = {'upper_body': ['torso_leftArm', 'torso_rightArm'],
                          'lower_body': ['torso_leftLeg', 'torso_rightLeg']}
        self.L4_groups = {'full_body': ['upper_body', 'lower_body', 'global_trans']}      
                                                  
        self.L1_units = 64
        self.L2_units = 128
        self.L3_units = 256
        self.L4_units = 512
        self.activation = tf.nn.elu
        self.layers_dict = OrderedDict()
        self.encoder_net = OrderedDict()
        self.decoder_net = OrderedDict()

        ### L1 group ###
        for group in self.L1_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L1_units, activation=self.activation, name=group)                       

        ### L2 group ###
        for group in self.L2_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L2_units, activation=self.activation, name=group)
        
        ### L3 group ###
        for group in self.L3_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L3_units, activation=self.activation, name=group)

        ### L4 group ###
        for group in self.L4_groups.keys():
            self.layers_dict[group] = layers.Dense(self.L4_units, activation=self.activation, name=group)    

        ### representation layer ###
        self.layers_dict['z_layer'] = layers.Dense(self.z_unit, activation=self.activation, name='z_layer')

        ### L4 inverse group ###
        for group in self.L4_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L4_units, activation=self.activation, name=group + '_inverse')            

        ### L3 inverse group ###
        for group in self.L3_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L3_units, activation=self.activation, name=group + '_inverse')

        ### L2 inverse group ###
        for group in self.L2_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L2_units, activation=self.activation, name=group + '_inverse')

        ### L1 inverse group ###
        for group in self.L1_groups.keys():
            self.layers_dict[group + '_inverse'] = layers.Dense(self.L1_units, activation=self.activation, name=group + '_inverse')

        ### output group ###
        for group in self.L1_groups.keys():
            if group == "global_trans":
                self.layers_dict[group + '_output'] = layers.Dense(3, name=group + '_output')
            else:
                self.layers_dict[group + '_output'] = layers.Dense(len(self.L1_groups[group]) * 3, name=group + '_output')   

        self.layers_dict['dropout'] = layers.Dropout(dropout_rate) 

    def _encoder(self, inputs, training=False):

        for group_name, values in self.L1_groups.items():
            ## preparing inputs
            if group_name == "global_trans":
                self.encoder_net[group_name + '_input'] = inputs[:, -3:]  ## assume the last three dimensions are global transformation
            else:
                self.encoder_net[group_name + '_input'] = tf.concat([inputs[:, self.animated_joints.index(joint)*self.num_params_per_joint:
                                                          (self.animated_joints.index(joint) + 1) * self.num_params_per_joint] for joint in values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])  
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)                                   

        ### encoder L2 ###
        for group_name, group_values in self.L2_groups.items():
            ### merge L1 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            # self.encoder_net[group_name] = self.layers_dict[group_name](input) 
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training) 
            # self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            # if training:
            #     self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L3 ###

        for group_name, group_values in self.L3_groups.items():
            ### merge L2 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            # self.encoder_net[group_name] = self.layers_dict[group_name](input)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)   
            # self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            # if training:
            #     self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L4 ###

        for group_name, group_values in self.L4_groups.items():
            ### merge L3 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            # if training:
            #     input = self.layers_dict['dropout'](self.encoder_net[group_name + '_input'], training=training)
            # else:
            #     input = self.encoder_net[group_name + '_input']
            # self.encoder_net[group_name] = self.layers_dict[group_name](input)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])
            self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)
            # self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            # if training:
            #     self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)

        ### create hidden representation
        # the input is fixed by the keyword 'full_body'
        # if training:
        #     self.encoder_net['full_body'] = self.layers_dict['dropout'](self.encoder_net['full_body'], training=training)
        # self.encoder_net['z_code'] = self.layers_dict['z_layer'](self.encoder_net['full_body'])

        self.encoder_net['z_code'] = self.layers_dict['z_layer'](self.encoder_net['full_body'])
        self.encoder_net['z_code'] = self.layers_dict['dropout'](self.encoder_net['z_code'], training=training)
        return self.encoder_net['z_code']

    def _decoder(self, inputs, training=False):
        ### decoder L4 ###
        # if training:
        #     inputs = self.layers_dict['dropout'](inputs, training=training)

        self.decoder_net['full_body'] = self.layers_dict['full_body_inverse'](inputs)
        self.decoder_net['full_body'] = self.layers_dict['dropout'](self.decoder_net['full_body'], training=training)
        
        ### decoder L3 ###

        for group in self.L3_groups.keys():
            ### super group can be more than one
            super_groups = [group_name for group_name, values in self.L4_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            # if training:
            #     layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])
                
        ### decoder L2 ###
        for group in self.L2_groups.keys():

            super_groups = [group_name for group_name, values in self.L3_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            # if training:
            #     layer_input = self.layers_dict['dropout'](layer_input, training=training)
            self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder L1 ###
        for group in self.L1_groups.keys():

            if group == "global_trans":
                # if training:
                #     self.decoder_net['full_body'] = self.layers_dict['dropout'](self.decoder_net['full_body'], training=training)
                # self.decoder_net[group] = self.layers_dict[group+'_inverse'](self.decoder_net['full_body'])
                self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](self.decoder_net['full_body']), training=training)
            else:
                super_groups = [group_name for group_name, values in self.L2_groups.items() if group in values]
                layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
                # if training:
                #     layer_input = self.layers_dict['dropout'](layer_input, training=training)
                # self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
                self.decoder_net[group] = self.layers_dict['dropout'](self.layers_dict[group+'_inverse'](layer_input), training=training)
            # if training:
            #     self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder output ###

        for group, joints in self.L1_groups.items():
            self.decoder_net[group + '_output'] = self.layers_dict[group + '_output'](self.decoder_net[group])

            for joint in joints:
                self.decoder_net[joint] = self.decoder_net[group + '_output'][:, joints.index(joint) * self.num_params_per_joint : (joints.index(joint) + 1) * self.num_params_per_joint]
        ### reorder 
        output = tf.concat([self.decoder_net[joint] for joint in self.animated_joints], axis=-1)
        output = tf.concat([output, self.decoder_net['global_trans_output']], axis=-1)
        return output
            
    def call(self, inputs, training=False):
        z_encoder = self._encoder(inputs, training=training)
        decoreded = self._decoder(z_encoder, training=training)
        return decoreded

    def fine_tuning(self, model_file, training_data, epochs, batchsize, learning_rate, save_path, save_every_epochs=None):
        ### load existing model
        self.load_weights(model_file)
        n_samples = len(training_data)
        n_batches = n_samples // batchsize
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        training_data = tf.data.Dataset.from_tensor_slices((training_data, training_data)).batch(batchsize)
        filename = self.name + "_fine_tuning_{epoch:04d}.ckpt"
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            progbar = tf.keras.utils.Progbar(n_batches)
            for i, (X_batch, Y_batch) in enumerate(training_data):
                with tf.GradientTape() as tape:
                    predict = self.call(X_batch)
                    Y_batch = tf.cast(Y_batch, tf.float32)
                    predict = tf.cast(predict, tf.float32)
                    loss = loss_global_position(Y_batch, predict)
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                progbar.update(i, [("loss", loss)])
            print("\n")
            if save_every_epochs is not None:
                if (epoch + 1) % save_every_epochs == 0:
                    self.save_weights(os.path.join(save_path, filename.format(epoch=(epoch+1))))
        ### save fine-tuned model
        self.save_weights(os.path.join(save_path, self.name + 'ckpt'))