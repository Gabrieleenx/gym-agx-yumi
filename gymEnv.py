
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
#print('FILE_DIR', FILE_DIR)
PACKAGE_DIR = FILE_DIR #os.path.split(FILE_DIR)[0]
#print('PACKAGE_DIR', PACKAGE_DIR)

URDF_PATH = FILE_DIR + "/yumi_description/urdf/yumi.urdf"
#print('URDF_PATH', URDF_PATH)

def collisionBetweenBodies(RB1, RB2, collision=True):
    for i in range(len(RB1.getGeometries())):
        for j in range(len(RB2.getGeometries())):
            RB1.getGeometries()[i].setEnableCollisions(RB2.getGeometries()[j], collision)


class YumiPegInHole(AGXGymEnv):
    def __init__(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps
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
        self.lastClosestDist = None
        self.oldNSegmentsInserted = 0
        self.goalThreshold = 5
        super().__init__()

    def _build_scene(self):

        scene = agxSDK.Assembly()  # Create a new empty Assembly
        if not agxIO.readFile(self.scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
            raise RuntimeError("Unable to open file \'" + self.scene_path + "\'")
        scene.setName("main_assembly")
        self.sim.add(scene)
        self.yumi = self.sim.getAssembly('yumi')
        self.floor = self.sim.getAssembly('floor')

        # contact event listener     
        self.sim.addEventListener(utils.contactEventListenerRigidBody('yumiContact', self.sim.getRigidBody('gripper_r_base')))
        self.sim.addEventListener(utils.contactEventListenerRigidBody('gripper_r_finger_r', self.sim.getRigidBody('gripper_r_finger_r')))
        self.sim.addEventListener(utils.contactEventListenerRigidBody('gripper_r_finger_l', self.sim.getRigidBody('gripper_r_finger_l')))



    def _modify_visuals(self, root):
        self.app.getSceneDecorator().setBackgroundColor(agxRender.Color.BlanchedAlmond(), agxRender.Color.DimGray())

        cameraData = self.app.getCameraData()
        cameraData.eye = agx.Vec3(3, -3, 1.0)
        cameraData.center = agx.Vec3(0, 0, 0)
        cameraData.up = agx.Vec3(0, 0, 1)
        cameraData.nearClippingPlane = 0.1
        cameraData.farClippingPlane = 5000
        self.app.applyCameraData(cameraData)

        self.agent_scene_decorator = AgentSceneDecorator(self.app)

        #fl = agxOSG.createVisual(self.floor, root)
        #agxOSG.setDiffuseColor(fl, agxRender.Color.LightGray())
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
        jointPositions = np.array(jointPositions, dtype=np.float32)
        jointVelocities = np.array(jointVelocities, dtype=np.float32)

        # meassure force and torque (seams to be in zyx)
        forces = [0,0,0,0,0,0]
        for i in range(6): 
            forces[i] = self.yumi.getConstraint('yumi_link_7_r_joint').getCurrentForce(i)
        
        # get DLO point cloud 
        self.DLOPointCloud = utils.compute_segments_pos(self.sim)
        DLOPointCloud = self.DLOPointCloud.flatten()

        # get Cylinder position 

        cylinder = self.sim.getRigidBody("hollow_cylinder")
        cylinderPos = cylinder.getPosition()
        self.cylinderPos = utils.to_numpy_array(cylinderPos)

        # stack observations 
        o = np.hstack([jointPositions, jointVelocities, DLOPointCloud, self.cylinderPos])
        
        t = self._terminal(o)
        r = self._reward(o, t)

        if self.app:
            try:
                self.agent_scene_decorator.update(self.step_nb, r, observation=o)
            except:
                pass

        return o, r, t, {}


    def _set_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i in range(len(self.jointNamesRevolute)):
            self.yumi.getConstraint1DOF(self.jointNamesRevolute[i]).getMotor1D().setSpeed(float(action[i]) )

    def _reward(self, o, t):
        #TODO update reward
        r = 0
        closestDist = self._closestPointToTarget()
        if self.lastClosestDist == None:
            self.lastClosestDist = closestDist

        if closestDist < self.lastClosestDist:
            r += 1e-2
        elif closestDist > self.lastClosestDist:
            r -= 1e-2
        else:
            r += 0

        nSegmentsInserted = self._determineNSegmentsInserted()
        diff= nSegmentsInserted - self.oldNSegmentsInserted
        self.oldNSegmentsInserted = nSegmentsInserted
        r += diff

        if self._is_goal_reached(nSegmentsInserted):
            return r + 5 

        if t:
            return -1

        return r

    def _terminal(self, o):
        return self.step_nb >= self.max_episode_steps or self._yumiCollision()

    def _yumiCollision(self):
        event1 = self.sim.getEventListener('yumiContact')
        event2 = self.sim.getEventListener('gripper_r_finger_r')
        event3 = self.sim.getEventListener('gripper_r_finger_l')
        
        return event1.contactState or event2.contactState or event3.contactState


    def _is_goal_reached(self, nInsertedSegments):
        if nInsertedSegments > self.goalThreshold:
            return True
        else:
            return False

    def _determineNSegmentsInserted(self):
        n_inserted = 0
        for i in range(self.DLOPointCloud.shape[0]):
            p = self.DLOPointCloud[i]
            if (self.cylinderPos[0]-0.015 <= p[0] <= self.cylinderPos[0]+0.015 and
                    self.cylinderPos[1]-0.015 <= p[1] <= self.cylinderPos[1]+0.015 and
                    -0.1 <= p[2] <=0.07):
                n_inserted += 1
        return n_inserted

    def _closestPointToTarget(self): 
        dist = np.linalg.norm(self.DLOPointCloud -self.cylinderPos, axis=1)
        return np.min(dist)
            


