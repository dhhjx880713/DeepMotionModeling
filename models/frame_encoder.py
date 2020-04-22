import tensorflow as tf 
import numpy as np
POINT_LEN = 3
from tensorflow.keras import layers, Model
from collections import OrderedDict
tf.keras.backend.set_floatx('float64')


class FrameEncoderNoGlobal(tf.keras.Model):
    def __init__(self):
        super(FrameEncoderNoGlobal, self).__init__()
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
                          'rightLeg': ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']}
        self.L2_groups = {'torso_leftArm': ['torso', 'leftArm'],
                          'torso_rightArm': ['torso', 'rightArm'],
                          'torso_leftLeg': ['torso', 'leftLeg'],
                          'torso_rightLeg': ['torso', 'rightLeg']}
        self.L3_groups = {'upper_body': ['torso_leftArm', 'torso_rightArm'],
                          'lower_body': ['torso_leftLeg', 'torso_rightLeg']}
        self.L4_groups = {'full_body': ['upper_body', 'lower_body']}                                                

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

        self.layers_dict['dropout'] = layers.Dropout(0.25) 

    def _encoder(self, inputs, training=False):

        for group_name, values in self.L1_groups.items():
            ## preparing inputs

            self.encoder_net[group_name + '_input'] = tf.concat([inputs[:, self.animated_joints.index(joint)*self.num_params_per_joint:
                                                          (self.animated_joints.index(joint) + 1) * self.num_params_per_joint] for joint in values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])                                     
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)

        ### encoder L2 ###
        for group_name, group_values in self.L2_groups.items():
            ### merge L1 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L3 ###

        for group_name, group_values in self.L3_groups.items():
            ### merge L2 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L4 ###

        for group_name, group_values in self.L4_groups.items():
            ### merge L3 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)

        ### create hidden representation
        # the input is fixed by the keyword 'full_body'
        self.encoder_net['z_code'] = self.layers_dict['z_layer'](self.encoder_net['full_body'])
        if training:
            self.encoder_net['z_code'] = self.layers_dict['dropout'](self.encoder_net['z_code'], training=training)
        return self.encoder_net['z_code']

    def _decoder(self, inputs, training=False):
        ### decoder L4 ###
        self.decoder_net['full_body'] = self.layers_dict['full_body_inverse'](inputs)
        if training:
            self.decoder_net['full_body'] = self.layers_dict['dropout'](self.decoder_net['full_body'])
        
        ### decoder L3 ###

        for group in self.L3_groups.keys():
            ### super group can be more than one
            super_groups = [group_name for group_name, values in self.L4_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            if training:
                self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])
                
        ### decoder L2 ###
        for group in self.L2_groups.keys():

            super_groups = [group_name for group_name, values in self.L3_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            if training:
                self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder L1 ###
        for group in self.L1_groups.keys():

            super_groups = [group_name for group_name, values in self.L2_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            if training:
                self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

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

class FrameEncoder(tf.keras.Model):
    """spaial motion encoder based on skeleton hierarchy
    
    """
    def __init__(self, dropout_rate):
        super(FrameEncoder, self).__init__()
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
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input'])                                     
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)

        ### encoder L2 ###
        for group_name, group_values in self.L2_groups.items():
            ### merge L1 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L3 ###

        for group_name, group_values in self.L3_groups.items():
            ### merge L2 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)            

        ### encoder L4 ###

        for group_name, group_values in self.L4_groups.items():
            ### merge L3 results
            self.encoder_net[group_name + '_input'] = tf.concat([self.encoder_net[group_value] for group_value in group_values], axis=-1)
            self.encoder_net[group_name] = self.layers_dict[group_name](self.encoder_net[group_name + '_input']) 
            if training:
                self.encoder_net[group_name] = self.layers_dict['dropout'](self.encoder_net[group_name], training=training)

        ### create hidden representation
        # the input is fixed by the keyword 'full_body'
        self.encoder_net['z_code'] = self.layers_dict['z_layer'](self.encoder_net['full_body'])
        if training:
            self.encoder_net['z_code'] = self.layers_dict['dropout'](self.encoder_net['z_code'], training=training)
        return self.encoder_net['z_code']

    def _decoder(self, inputs, training=False):
        ### decoder L4 ###
        self.decoder_net['full_body'] = self.layers_dict['full_body_inverse'](inputs)
        if training:
            self.decoder_net['full_body'] = self.layers_dict['dropout'](self.decoder_net['full_body'])
        
        ### decoder L3 ###

        for group in self.L3_groups.keys():
            ### super group can be more than one
            super_groups = [group_name for group_name, values in self.L4_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            if training:
                self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])
                
        ### decoder L2 ###
        for group in self.L2_groups.keys():

            super_groups = [group_name for group_name, values in self.L3_groups.items() if group in values]
            layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
            self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            if training:
                self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder L1 ###
        for group in self.L1_groups.keys():
            if group == "global_trans":
                self.decoder_net[group] = self.layers_dict[group+'_inverse'](self.decoder_net['full_body'])
            else:
                super_groups = [group_name for group_name, values in self.L2_groups.items() if group in values]
                layer_input = tf.add_n([self.decoder_net[super_group] for super_group in super_groups])
                self.decoder_net[group] = self.layers_dict[group+'_inverse'](layer_input)
            if training:
                self.decoder_net[group] = self.layers_dict['dropout'](self.decoder_net[group])

        ### decoder output ###

        for group, joints in self.L1_groups.items():
            self.decoder_net[group + '_output'] = self.layers_dict[group + '_output'](self.decoder_net[group])
            if group != 'global_trans':
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


