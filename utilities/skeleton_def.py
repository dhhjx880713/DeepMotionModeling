import collections


GAME_ENGINE_ANIMATED_JOINTS = ['Game_engine', 'Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l', 'lowerarm_l',
                               'hand_l', 'clavicle_r',
                               'upperarm_r', 'lowerarm_r', 'hand_r', 'neck_01', 'head', 'thigh_l', 'calf_l', 'foot_l',
                               'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']


GAME_ENGINE_ANIMATED_JOINTS_without_game_engine = ['Root', 'pelvis', 'spine_03', 'clavicle_l', 'upperarm_l', 'lowerarm_l',
                                                   'hand_l', 'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r',
                                                   'neck_01', 'head', 'thigh_l', 'calf_l', 'foot_l',
                                                   'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r']


Edinburgh_animated_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg',
                             'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck1', 'Head', 'LeftArm', 'LeftForeArm',
                             'LeftHand', 'LeftHandIndex1', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex1']


MH_CMU_ANIMATED_JOINTS = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LowerBack', 'Spine',
                          'Spine1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LThumb', 'LeftFingerBase',
                          'LeftHandFinger1', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
                          'RightHand', 'RThumb', 'RightFingerBase', 'RightHandFinger1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']


### LHipJoint and RHipJoint are removed since their body length are 0
MH_CMU_SKELETON = collections.OrderedDict(  
    [
        ('Hips', {'parent': None, 'index': 0}),
        ('LeftUpLeg', {'parent': 'Hips', 'index': 1}),
        ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 2}),
        ('LeftFoot', {'parent': 'LeftLeg', 'index': 3}),
        ('LeftToeBase', {'parent': 'LeftFoot', 'index': 4}),
        ('LowerBack', {'parent': 'Hips', 'index': 5}),
        ('Spine', {'parent': 'LowerBack', 'index': 6}),
        ('Spine1', {'parent': 'Spine', 'index': 7}),
        ('LeftShoulder', {'parent': 'Spine1', 'index': 8}),
        ('LeftArm', {'parent': 'LeftShoulder', 'index': 9}),
        ('LeftForeArm', {'parent': 'LeftArm', 'index': 10}),
        ('LeftHand', {'parent': 'LeftForeArm', 'index': 11}),
        ('LThumb', {'parent': 'LeftHand', 'index': 12}),
        ('LeftFingerBase', {'parent': 'LeftHand', 'index': 13}),
        ('LeftHandFinger1', {'parent': 'LeftFingerBase', 'index': 14}),
        ('Neck', {'parent': 'Spine1', 'index': 15}),
        ('Neck1', {'parent': 'Neck', 'index': 16}),
        ('Head', {'parent': 'Neck1', 'index': 17}),
        ('RightShoulder', {'parent': 'Spine1', 'index': 18}),
        ('RightArm', {'parent': 'RightShoulder', 'index': 19}),
        ('RightForeArm', {'parent': 'RightArm', 'index': 20}),
        ('RightHand', {'parent': 'RightForeArm', 'index': 21}),
        ('RThumb', {'parent': 'RightHand', 'index': 22}),
        ('RightFingerBase', {'parent': 'RightHand', 'index': 23}),
        ('RightHandFinger1', {'parent': 'RightFingerBase', 'index': 24}),
        ('RightUpLeg', {'parent': 'Hips', 'index': 25}),
        ('RightLeg', {'parent': 'RightUpLeg', 'index': 26}),
        ('RightFoot', {'parent': 'RightLeg', 'index': 27}),
        ('RightToeBase', {'parent': 'RightFoot', 'index': 28})
    ]
)


MH_CMU_SKELETON_FULL = collections.OrderedDict(  
    [
        ('Hips', {'parent': None, 'index': 0}),
        ('LHipJoint', {'parent': 'Hips', 'index': 1}),  
        ('LeftUpLeg', {'parent': 'Hips', 'index': 2}),
        ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 3}),
        ('LeftFoot', {'parent': 'LeftLeg', 'index': 4}),
        ('LeftToeBase', {'parent': 'LeftFoot', 'index': 5}),
        ('LowerBack', {'parent': 'Hips', 'index': 6}),
        ('Spine', {'parent': 'LowerBack', 'index': 7}),
        ('Spine1', {'parent': 'Spine', 'index': 8}),
        ('LeftShoulder', {'parent': 'Spine1', 'index': 9}),
        ('LeftArm', {'parent': 'LeftShoulder', 'index': 10}),
        ('LeftForeArm', {'parent': 'LeftArm', 'index': 11}),
        ('LeftHand', {'parent': 'LeftForeArm', 'index': 12}),
        ('LThumb', {'parent': 'LeftHand', 'index': 13}),
        ('LeftFingerBase', {'parent': 'LeftHand', 'index': 14}),
        ('LeftHandFinger1', {'parent': 'LeftFingerBase', 'index': 15}),
        ('Neck', {'parent': 'Spine1', 'index': 16}),
        ('Neck1', {'parent': 'Neck', 'index': 17}),
        ('Head', {'parent': 'Neck1', 'index': 18}),
        ('RightShoulder', {'parent': 'Spine1', 'index': 19}),
        ('RightArm', {'parent': 'RightShoulder', 'index': 20}),
        ('RightForeArm', {'parent': 'RightArm', 'index': 21}),
        ('RightHand', {'parent': 'RightForeArm', 'index': 22}),
        ('RThumb', {'parent': 'RightHand', 'index': 23}),
        ('RightFingerBase', {'parent': 'RightHand', 'index': 24}),
        ('RightHandFinger1', {'parent': 'RightFingerBase', 'index': 25}),
        ('RHipJoint', {'parent': 'Hips', 'index': 26}),  # 5
        ('RightUpLeg', {'parent': 'Hips', 'index': 27}),
        ('RightLeg', {'parent': 'RightUpLeg', 'index': 28}),
        ('RightFoot', {'parent': 'RightLeg', 'index': 29}),
        ('RightToeBase', {'parent': 'RightFoot', 'index': 30})
    ]
)




GAME_ENGINE_SKELETON = collections.OrderedDict(
    [
        ('Root', {'parent': None, 'index': 0}),
        ('pelvis', {'parent': 'Root', 'index': 1}),
        ('spine_03', {'parent': 'pelvis', 'index': 2}),
        ('clavicle_l', {'parent': 'spine_03', 'index': 3}),
        ('upperarm_l', {'parent': 'clavicle_l', 'index': 4}),
        ('lowerarm_l', {'parent': 'upperarm_l', 'index': 5}),
        ('hand_l', {'parent': 'lowerarm_l', 'index': 6}),
        ('clavicle_r', {'parent': 'spine_03', 'index': 7}),
        ('upperarm_r', {'parent': 'clavicle_r', 'index': 8}),
        ('lowerarm_r', {'parent': 'upperarm_r', 'index': 9}),
        ('hand_r', {'parent': 'lowerarm_r', 'index': 10}),
        ('neck_01', {'parent': 'spine_03', 'index': 11}),
        ('head', {'parent': 'neck_01', 'index': 12}),
        ('thigh_l', {'parent': 'pelvis', 'index': 13}),
        ('calf_l', {'parent': 'thigh_l', 'index': 14}),
        ('foot_l', {'parent': 'calf_l', 'index': 15}),
        ('ball_l', {'parent': 'foot_l', 'index': 16}),
        ('thigh_r', {'parent': 'pelvis', 'index': 17}),
        ('calf_r', {'parent': 'thigh_r', 'index': 18}),
        ('foot_r', {'parent': 'calf_r', 'index': 19}),
        ('ball_r', {'parent': 'foot_r', 'index': 20})
    ]
)

Edinburgh_skeleton = collections.OrderedDict(
    [
        ('Root', {'parent': None, 'index': 0}),
        ('Hips', {'parent': 'Root', 'index': 1}),
        ('LeftUpLeg', {'parent': 'Hips', 'index': 2}),
        ('LeftLeg', {'parent': 'LeftUpLeg', 'index': 3}),
        ('LeftFoot', {'parent': 'LeftLeg', 'index': 4}),
        ('LeftToeBase', {'parent': 'LeftFoot', 'index': 5}),
        ('RightUpLeg', {'parent': 'Hips', 'index': 6}),
        ('RightLeg', {'parent': 'RightUpLeg', 'index': 7}),
        ('RightFoot', {'parent': 'RightLeg', 'index': 8}),
        ('RightToeBase', {'parent': 'RightFoot', 'index': 9}),
        ('Spine', {'parent': 'Hips', 'index': 10}),
        ('Spine1', {'parent': 'Spine', 'index': 11}),
        ('Neck1', {'parent': 'Spine1', 'index': 12}),
        ('Head', {'parent': 'Neck1', 'index': 13}),
        ('LeftArm', {'parent': 'Spine1', 'index': 14}),
        ('LeftForeArm', {'parent': 'LeftArm', 'index': 15}),
        ('LeftHand', {'parent': 'LeftForeArm', 'index': 16}),
        ('LeftHandIndex1', {'parent': 'LeftHand', 'index': 17}),
        ('RightArm', {'parent': 'Spine1', 'index': 18}),
        ('RightForeArm', {'parent': 'RightArm', 'index': 19}),
        ('RightHand', {'parent': 'RightForeArm', 'index': 20}),
        ('RightHandIndex1', {'parent': 'RightHand', 'index': 21})
    ]
)
