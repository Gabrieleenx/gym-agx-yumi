
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxSDK
import agxCable
import agxIO
import agxOSG
import agxRender
import agxModel
# Python modules
import sys
#import logging
import numpy as np
import os

import agxUtil

import utils

#logger = logging.getLogger('gym_agx.sims')
# Set paths
FILE_NAME = 'yumi_test'
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
#print('FILE_DIR', FILE_DIR)
PACKAGE_DIR = FILE_DIR #os.path.split(FILE_DIR)[0]
#print('PACKAGE_DIR', PACKAGE_DIR)

URDF_PATH = FILE_DIR + "/yumi_description/urdf/yumi.urdf"
#print('URDF_PATH', URDF_PATH)

#FILE_NAME = "peg_in_hole"
# Simulation parameters
#TIMESTEP = 1 / 500
TIMESTEP = 1 / 50
N_SUBSTEPS = 20
GRAVITY = True
# Rope parameters
RADIUS = 0.005  # meters
RESOLUTION = 100  # segments per meter
PEG_POISSON_RATIO = 0.1  # no unit
YOUNG_MODULUS_BEND = 2e5  # 1e4
YOUNG_MODULUS_TWIST = 2e5  # 1e10
YOUNG_MODULUS_STRETCH = 1e8  # Pascals
ROPE_LENGTH = 0.2
# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals
# Meshes and textures
MESH_HOLLOW_CYLINDER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_hollow_cylinder.obj")
# Ground Parameters
EYE = agx.Vec3(0, -1, 0.2)
CENTER = agx.Vec3(0, 0, 0.2)
UP = agx.Vec3(0., 0., 1.0)

# Control parameters
JOINT_RANGES = {"t_x": [-0.1,0.1],
                "t_y": [-0.05,0.05],
                "t_z": [-0.15,0.05],
                "r_y": [(-1/4)*np.pi,(1/4)*np.pi]}
FORCE_RANGES = {"t_x": [-5,5], "t_y": [-5,5], "t_z": [-5,5], "r_y": [-5,5]}


def create_yumi(sim):
    initJointPosList = [0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 0.0, 0.0, -0.7, -1.7, 0.8, 1.0, 2.2, 1.0, 0.0, 0.0, 0.0]
    # urdf name for joints 
    jointNamesRevolute = ['yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l',\
                                'yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r']
    
    jointEffort = [50,50,50,50,50,50,50,50,50,50,50,50,50,50] #maximum joint effort, assuming same force in upper and lower, same order as jointNamesRevolute
    grpperEffort  = 15 # set the grip force
    jointNamesGrippers = ['gripper_l_joint', 'gripper_l_joint_m', 'gripper_r_joint', 'gripper_r_joint_m'] # name of gripper joints in urdf
    gripperPosition = [0,0,0,0] # used to store gripper commands until they are used
    gripperPositionRun = [0,0,0,0] # uesd for controlling. Both are needed for emulating yumi behaviour 

    # ------------- YuMi -------------------------------------------------- 

    # initial joint position 
    initJointPos = agx.RealVector()
    for i in range(len(initJointPosList)):
        initJointPos.append(initJointPosList[i])

    # read urdf
    yumi_assembly_ref = agxModel.UrdfReader.read(URDF_PATH, PACKAGE_DIR, initJointPos, True)
    if (yumi_assembly_ref.get() == None):
        print("Error reading the URDF file.")
        sys.exit(2)
    
    # Add the yumi assembly to the simulation and create visualization for it
    sim.add(yumi_assembly_ref.get())
    
    yumi = yumi_assembly_ref.get()

    # Enable Motor1D (speed controller) on all revolute joints and set effort limits 
    for i in range(len(jointNamesRevolute)):
        yumi.getConstraint1DOF(jointNamesRevolute[i]).getMotor1D().setEnable(True)
        yumi.getConstraint1DOF(jointNamesRevolute[i]).getMotor1D().setForceRange(-jointEffort[i], jointEffort[i])
    
    # Enable Motor1D (speed controller) on all prismatic joints (grippers) and set effort limits 
    for i in range(len(jointNamesGrippers)):
        yumi.getConstraint1DOF(jointNamesGrippers[i]).getMotor1D().setEnable(True)
        yumi.getConstraint1DOF(jointNamesGrippers[i]).getMotor1D().setForceRange(-grpperEffort, grpperEffort)

    # disable collision between connected links. 
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_body'), yumi_assembly_ref.getRigidBody('yumi_link_1_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_body'), yumi_assembly_ref.getRigidBody('yumi_link_1_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_1_r'), yumi_assembly_ref.getRigidBody('yumi_link_2_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_1_l'), yumi_assembly_ref.getRigidBody('yumi_link_2_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_2_r'), yumi_assembly_ref.getRigidBody('yumi_link_3_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_2_l'), yumi_assembly_ref.getRigidBody('yumi_link_3_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_3_r'), yumi_assembly_ref.getRigidBody('yumi_link_4_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_3_l'), yumi_assembly_ref.getRigidBody('yumi_link_4_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_4_r'), yumi_assembly_ref.getRigidBody('yumi_link_5_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_4_l'), yumi_assembly_ref.getRigidBody('yumi_link_5_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_5_r'), yumi_assembly_ref.getRigidBody('yumi_link_6_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_5_l'), yumi_assembly_ref.getRigidBody('yumi_link_6_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_r'), yumi_assembly_ref.getRigidBody('yumi_link_7_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_l'), yumi_assembly_ref.getRigidBody('yumi_link_7_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_r'), yumi_assembly_ref.getRigidBody('yumi_link_7_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_6_l'), yumi_assembly_ref.getRigidBody('yumi_link_7_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_7_r'), yumi_assembly_ref.getRigidBody('gripper_r_base'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('yumi_link_7_l'), yumi_assembly_ref.getRigidBody('gripper_l_base'), False)
    
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_r_base'), yumi_assembly_ref.getRigidBody('gripper_r_finger_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_r_base'), yumi_assembly_ref.getRigidBody('gripper_r_finger_l'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_r_finger_r'), yumi_assembly_ref.getRigidBody('gripper_r_finger_l'), False)

    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_l_base'), yumi_assembly_ref.getRigidBody('gripper_l_finger_r'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_l_base'), yumi_assembly_ref.getRigidBody('gripper_l_finger_l'), False)
    utils.collisionBetweenBodies(yumi_assembly_ref.getRigidBody('gripper_l_finger_r'), yumi_assembly_ref.getRigidBody('gripper_l_finger_l'), False)

def create_floor(sim):
    # ------------ Floor --------------------------------------------------
    # Construct the floor that the yumi robot will stand 
    
    #floor = agxCollide.Geometry(agxCollide.Box(agx.Vec3(1, 1, 0.05)))
    #floor.setPosition(0, 0, -0.05)
    #sim.add(floor)
    
    material_ground = agx.Material("Aluminum")
    floor = utils.create_body(name="floor", shape=agxCollide.Box(agx.Vec3(1, 1, 0.05)),
                        position=agx.Vec3(0, 0, -0.05),
                        motion_control=agx.RigidBody.STATIC,
                        material=material_ground)
    sim.add(floor)

def create_cylinder(sim, material_hard):
    # Create hollow cylinde with hole
    scaling_cylinder = agx.Matrix3x3(agx.Vec3(0.0275))
    reader = agxIO.MeshReader()
    hullMesh = agxUtil.createTrimesh(MESH_HOLLOW_CYLINDER_FILE,0, scaling_cylinder)
    #print(hullMesh)
    hullGeom = agxCollide.Geometry(hullMesh,  agx.AffineMatrix4x4.rotate(agx.Vec3(0,1,0),agx.Vec3(0,0,1)))
    hollow_cylinder = agx.RigidBody("hollow_cylinder")
    hollow_cylinder.add(hullGeom)
    hollow_cylinder.setMotionControl(agx.RigidBody.STATIC)
    hullGeom.setMaterial(material_hard)
    hollow_cylinder.setPosition(agx.Vec3(0.3, 0, 0.0))
    sim.add(hollow_cylinder)

def create_DLO(sim):
    # Create rope and set name + properties
    peg = agxCable.Cable(RADIUS, RESOLUTION)
    peg.setName("DLO")
    material_peg = peg.getMaterial()
    peg_material = material_peg.getBulkMaterial()
    peg_material.setPoissonsRatio(PEG_POISSON_RATIO)
    properties = peg.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

    # Add connection between cable and gripper
    tf_0 = agx.AffineMatrix4x4()
    tf_0.setTranslate(0.0 ,0, 0.135)
    peg.add(agxCable.BodyFixedNode(sim.getRigidBody("gripper_r_base"), tf_0))
    freePos = sim.getRigidBody("gripper_r_base").getTransform().transformPoint(agx.Vec3(0.0, 0, ROPE_LENGTH+0.135))
    peg.add(agxCable.FreeNode(freePos))

    sim.add(peg)
    
    segment_iterator = peg.begin()
    n_segments = peg.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            seg = segment_iterator.getRigidBody()
            seg.setAngularVelocityDamping(1e3)
            segment_iterator.inc()
    
    # Try to initialize rope
    report = peg.tryInitialize()
    if report.successful():
        print("Successful rope initialization.")
    else:
        print(report.getActualError())
    
    # Add rope to simulation
    #sim.add(peg)
    

    # Set rope material
    material_peg = peg.getMaterial()
    material_peg.setName("rope_material")
    
    material_hard = agx.Material("Aluminum")

    contactMaterial = sim.getMaterialManager().getOrCreateContactMaterial(material_hard, material_peg)
    contactMaterial.setYoungsModulus(1e12)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contactMaterial.setFrictionModel(fm)
    
    rbs = peg.getRigidBodies()
    for i in range(len(rbs)):
        rbs[i].setName('dlo_' + str(i+1))


def build_simulation():
    # Instantiate a simulation
    sim = agxSDK.Simulation()

    # By default the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
    # too by creating an agx.PointGravityField for example).
    # AGX uses a right-hand coordinate system (That is Z defines UP. X is right, and Y is into the screen)
    if not GRAVITY:
        #logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # Get current delta-t (timestep) that is used in the simulation?
    dt = sim.getTimeStep()
    #logger.debug("default dt = {}".format(dt))

    # Change the timestep
    sim.setTimeStep(TIMESTEP)

    # Confirm timestep changed
    dt = sim.getTimeStep()
    #logger.debug("new dt = {}".format(dt))

    # Define materials
    material_hard = agx.Material("Aluminum")
    material_hard_bulk = material_hard.getBulkMaterial()
    material_hard_bulk.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    material_hard_bulk.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)

    # create 
    create_floor(sim)
    create_yumi(sim)
    create_cylinder(sim, material_hard)
    create_DLO(sim)
    # disable collision between floor and yumi body
    utils.collisionBetweenBodies(sim.getAssembly('floor').getRigidBody('floor'), sim.getAssembly('yumi').getRigidBody('yumi_body'), False)
    dlo = agxCable.Cable.find(sim, "DLO")
    agxUtil.setEnableCollisions(dlo,  sim.getAssembly('yumi').getRigidBody('gripper_r_finger_r'), False)
    agxUtil.setEnableCollisions(dlo,  sim.getAssembly('yumi').getRigidBody('gripper_r_finger_l'), False)

    return sim



def main(args):
    # Build simulation object
    sim = build_simulation()

    # Save simulation to file
    success = utils.save_simulation(sim, FILE_NAME)
    #if success:
        #logger.debug("Simulation saved!")
    #else:
        #logger.debug("Simulation not saved!")
 
if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
    