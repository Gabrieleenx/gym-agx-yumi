
import gym

import math
import argparse
import numpy as np

# AGX Dynamics imports
import agx
import agxOSG
import agxUtil
import agxRender
import agxCollide
import os
import agxModel
import sys
import agxSDK
import agxIO

import utils

from agxPythonModules.agxGym.agx_env import AGXGymEnv
from agxPythonModules.agxGym.utils import AgentSceneDecorator
from agxPythonModules.agxGym.pfrl_utils import run_environment

# Set paths
FILE_NAME = 'yumi_test'
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
print('FILE_DIR', FILE_DIR)
PACKAGE_DIR = FILE_DIR #os.path.split(FILE_DIR)[0]
print('PACKAGE_DIR', PACKAGE_DIR)

URDF_PATH = FILE_DIR + "/yumi_description/urdf/yumi.urdf"
print('URDF_PATH', URDF_PATH)

def collisionBetweenBodies(RB1, RB2, collision=True):
    for i in range(len(RB1.getGeometries())):
        for j in range(len(RB2.getGeometries())):
            RB1.getGeometries()[i].setEnableCollisions(RB2.getGeometries()[j], collision)


class YumiPegInHole(AGXGymEnv):
    def __init__(self):
        self.max_episode_steps = 400
        self.initJointPosList = [1.0, -2.0, -1.2, 0.6, -2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, 1.2, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]
        # urdf name for joints 
        self.jointNamesRevolute = ['yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l',\
                                 'yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r']
       
        self.jointEffort = [50,50,50,50,50,50,50,50,50,50,50,50,50,50] #maximum joint effort, assuming same force in upper and lower, same order as jointNamesRevolute
        self.grpperEffort  = 15 # set the grip force
        self.jointNamesGrippers = ['gripper_l_joint', 'gripper_l_joint_m', 'gripper_r_joint', 'gripper_r_joint_m'] # name of gripper joints in urdf
        self.gripperPosition = [0,0,0,0] # used to store gripper commands until they are used
        self.gripperPositionRun = [0,0,0,0] # uesd for controlling. Both are needed for emulating yumi behaviour 
        self.scene_path = '/home/gabriel/gym-agx-yumi/assets/yumi_test.agx'
        super().__init__()

    def _build_scene(self):
        # ------------ Floor --------------------------------------------------
        # Construct the floor that the yumi robot will stand 
        
        #self.floor = agxCollide.Geometry(agxCollide.Box(agx.Vec3(1, 1, 0.05)))
        #self.floor.setPosition(0, 0, -0.05)
        #self.sim.add(self.floor)
        '''
        material_ground = agx.Material("Aluminum")
        self.floor = utils.create_body(name="floor", shape=agxCollide.Box(agx.Vec3(1, 1, 0.05)),
                            position=agx.Vec3(0, 0, -0.05),
                            motion_control=agx.RigidBody.STATIC,
                            material=material_ground)
        self.sim.add(self.floor)

        # ------------- YuMi -------------------------------------------------- 

        # initial joint position 
        initJointPos = agx.RealVector()
        for i in range(len(self.initJointPosList)):
            initJointPos.append(self.initJointPosList[i])

        # read urdf
        yumi_assembly_ref = agxModel.UrdfReader.read(URDF_PATH, PACKAGE_DIR, initJointPos, True)
        if (yumi_assembly_ref.get() == None):
            print("Error reading the URDF file.")
            sys.exit(2)
        
        # Add the yumi assembly to the simulation and create visualization for it
        self.sim.add(yumi_assembly_ref.get())
        
        self.yumi = yumi_assembly_ref.get()

        # Enable Motor1D (speed controller) on all revolute joints and set effort limits 
        for i in range(len(self.jointNamesRevolute)):
            self.yumi.getConstraint1DOF(self.jointNamesRevolute[i]).getMotor1D().setEnable(True)
            self.yumi.getConstraint1DOF(self.jointNamesRevolute[i]).getMotor1D().setForceRange(-self.jointEffort[i], self.jointEffort[i])
        
        # Enable Motor1D (speed controller) on all prismatic joints (grippers) and set effort limits 
        for i in range(len(self.jointNamesGrippers)):
            self.yumi.getConstraint1DOF(self.jointNamesGrippers[i]).getMotor1D().setEnable(True)
            self.yumi.getConstraint1DOF(self.jointNamesGrippers[i]).getMotor1D().setForceRange(-self.grpperEffort, self.grpperEffort)

        # disable collision between floor and yumi body
        #for i in range(len(yumi_assembly_ref.getRigidBody('yumi_body').getGeometries())):
        #    self.floor.getRigidBody('yumi_body').getGeometries()[0](yumi_assembly_ref.getRigidBody('yumi_body').getGeometries()[i], False)
        #print(self.floor.getRigidBody('floor'))
        collisionBetweenBodies(self.floor.getRigidBody('floor'), yumi_assembly_ref.getRigidBody('yumi_body'), False)

        # disable collision between connected links. 
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_body'), yumi_assembly_ref.getRigidBody('yumi_link_1_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_body'), yumi_assembly_ref.getRigidBody('yumi_link_1_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_1_r'), yumi_assembly_ref.getRigidBody('yumi_link_2_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_1_l'), yumi_assembly_ref.getRigidBody('yumi_link_2_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_2_r'), yumi_assembly_ref.getRigidBody('yumi_link_3_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_2_l'), yumi_assembly_ref.getRigidBody('yumi_link_3_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_3_r'), yumi_assembly_ref.getRigidBody('yumi_link_4_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_3_l'), yumi_assembly_ref.getRigidBody('yumi_link_4_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_4_r'), yumi_assembly_ref.getRigidBody('yumi_link_5_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_4_l'), yumi_assembly_ref.getRigidBody('yumi_link_5_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_5_r'), yumi_assembly_ref.getRigidBody('yumi_link_6_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_5_l'), yumi_assembly_ref.getRigidBody('yumi_link_6_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_r'), yumi_assembly_ref.getRigidBody('yumi_link_7_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_l'), yumi_assembly_ref.getRigidBody('yumi_link_7_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_r'), yumi_assembly_ref.getRigidBody('yumi_link_7_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_l'), yumi_assembly_ref.getRigidBody('yumi_link_7_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_7_r'), yumi_assembly_ref.getRigidBody('gripper_r_base'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_7_l'), yumi_assembly_ref.getRigidBody('gripper_l_base'), False)
        
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_r_base'), yumi_assembly_ref.getRigidBody('gripper_r_finger_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_r_base'), yumi_assembly_ref.getRigidBody('gripper_r_finger_l'), False)

        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_l_base'), yumi_assembly_ref.getRigidBody('gripper_l_finger_r'), False)
        collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_l_base'), yumi_assembly_ref.getRigidBody('gripper_l_finger_l'), False)
        
        # --------------- Task -------------------------------------------
        success = utils.save_simulation(self.sim, FILE_NAME)
        '''
        scene = agxSDK.Assembly()  # Create a new empty Assembly
        if not agxIO.readFile(self.scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
            raise RuntimeError("Unable to open file \'" + self.scene_path + "\'")
        scene.setName("main_assembly")
        self.sim.add(scene)
        print(type(self.sim))
        self.yumi = self.sim.getAssembly('yumi')
        self.floor = self.sim.getAssembly('floor')
        print('Finnish building ')
        #for j in assemblies:
        #    print(j.getName())
        #self.gravity = self.sim.getUniformGravity()
        #self.time_step = self.sim.getTimeStep()
        #logger.debug("Timestep after readFile is: {}".format(self.time_step))
        #logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def _modify_visuals(self, root):
        #print('Inside _modify_visuals')
        self.app.getSceneDecorator().setBackgroundColor(agxRender.Color.BlanchedAlmond(), agxRender.Color.DimGray())

        cameraData = self.app.getCameraData()
        cameraData.eye = agx.Vec3(3, -3, 1.0)
        cameraData.center = agx.Vec3(0, 0, 0)
        cameraData.up = agx.Vec3(0, 0, 1)
        cameraData.nearClippingPlane = 0.1
        cameraData.farClippingPlane = 5000
        self.app.applyCameraData(cameraData)

        self.agent_scene_decorator = AgentSceneDecorator(self.app)

        fl = agxOSG.createVisual(self.floor, root)
        agxOSG.setDiffuseColor(fl, agxRender.Color.LightGray())
        fl = agxOSG.createVisual(self.yumi, root)
        
    def _setup_gym_environment_spaces(self):
        # TODO update these when obervation changes. 
        self.action_space = gym.spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=np.float32), high=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]), shape=(16,), dtype=np.float32)
        upper = np.array([168.5, 43.5, 168.5, 80, 290, 138, 229], dtype=np.float32)*np.pi/(180)
        lower = np.array([-168.5, -143.5, -168.5, -123.5, -290, -88, -229], dtype=np.float32)*np.pi/(180)
        self.observation_space = gym.spaces.Box(
            low=np.hstack([upper, upper]),
            high=np.hstack([lower, lower]),
            shape=(14,),
            dtype=np.float32
        )


    def _observe(self):
        # TODO change observation
        jointPositions = []
        jointVelocities = []
        for i in range(len(self.jointNamesRevolute)):
            jointPositions.append(self.yumi.getConstraint1DOF(self.jointNamesRevolute[i]).getAngle())
            jointVelocities.append(self.yumi.getConstraint1DOF(self.jointNamesRevolute[i]).getCurrentSpeed())

        # meassure force and torque (seams to be in zyx)
        forces = [0,0,0,0,0,0]
        for i in range(6): 
            forces[i] = self.yumi.getConstraint('yumi_link_7_r_joint').getCurrentForce(i)

        DLOPointCloud = utils.compute_segments_pos(self.sim)

        o = np.array(jointPositions, dtype=np.float32)
        
        t = self._terminal(o)
        r = self._reward(o, t)

        if self.app:
            try:
                self.agent_scene_decorator.update(self.step_nb, r, observation=o)
            except:
                pass
                #print('-')


        return o, r, t, {}


    def _set_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print('action', action)
        for i in range(len(self.jointNamesRevolute)):
            self.yumi.getConstraint1DOF(self.jointNamesRevolute[i]).getMotor1D().setSpeed(float(action[i]) )

        #self.cart.addForce(20 * float(action[0]), 0.0, 0.0)

    def _reward(self, o, t):
        if t:
            return -1
        r = 0
        #if abs(o[2]) < math.radians(12):
        #    r = 1
        return r

    def _terminal(self, o):
        #o_clipped = np.clip(o, self.observation_space.low, self.observation_space.high)
        #crashed = not np.array_equiv(o_clipped, o)
        return self.step_nb >= self.max_episode_steps# or crashed

