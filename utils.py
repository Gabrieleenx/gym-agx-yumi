import agx
import agxIO
import agxSDK
import agxCollide
import agxCable

import os
import math
import numpy as np

def save_simulation(sim, file_name):
    """Save AGX simulation object to file.
    :param agxSDK.Simulation sim: AGX simulation object
    :param str file_name: name of the file
    :return: Boolean for success/failure
    """
    file_directory = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.split(file_directory)[0]
    markup_file = os.path.join(file_directory, 'assets', file_name + ".aagx")
    #agxIO.writeFile(markup_file, sim)
    #if not agxIO.writeFile(markup_file, sim):
    #    print("Unable to save simulation to markup file!")
    #    return False
    binary_file = os.path.join(file_directory, 'assets', file_name + ".agx")
    if not agxIO.writeFile(binary_file, sim):
        print("Unable to save simulation to binary file!")
        return False
    return True


def create_body(shape, name="", position=agx.Vec3(0, 0, 0), rotation=agx.OrthoMatrix3x3(),
                geometry_transform=agx.AffineMatrix4x4(), motion_control=agx.RigidBody.DYNAMICS, material=None,
                disable_collisions=False):
    """Helper function that creates a RigidBody according to the given definition.
    Returns the body itself, it's geometry and the OSG node that was created for it.
    :param agxCollide.Shape shape: shape of object.
    :param string name: Optional. Defaults to "". The name of the new body.
    :param agx.Vec3 position: Position of the object in world coordinates.
    :param agx.OrthoMatrix3x3 rotation: Rotation of the object in world coordinate frames
    :param agx.AffineMatrix4x4 geometry_transform: Optional. Defaults to identity transformation. The local
    transformation of the shape relative to the body.
    :param agx.RigidBody.MotionControl motion_control: Optional. Defaults to DYNAMICS.
    :param agx.Material material: Optional. Ignored if not given. Material assigned to the geometry created for the
    body.
    :param bool disable_collisions: Optional. Disable geometry collisions.
    :return: assembly
    """
    assembly = agxSDK.Assembly()
    assembly.setName(name)
    try:
        body = agx.RigidBody(name)
        geometry = agxCollide.Geometry(shape)
        geometry.setName(name)

        if disable_collisions:
            geometry.setEnableCollisions(False)

        if material:
            geometry.setMaterial(material)

        body.add(geometry, geometry_transform)
        body.setMotionControl(motion_control)
        assembly.add(body)
        assembly.setPosition(position)
        assembly.setRotation(rotation)

    except Exception as exception:
        print('failure to create body')
        #logger.error(exception)
    finally:
        return assembly


def collisionBetweenBodies(RB1, RB2, collision=True):
    for i in range(len(RB1.getGeometries())):
        for j in range(len(RB2.getGeometries())):
            RB1.getGeometries()[i].setEnableCollisions(RB2.getGeometries()[j], collision)

def to_numpy_array(agx_list):
    """Convert from AGX data structure to NumPy array.
    :param agx_list: AGX data structure
    :return: NumPy array
    """
    agx_type = type(agx_list)
    if agx_type == agx.Vec3:
        np_array = np.zeros(shape=(3,), dtype=np.float64)
        for i in range(3):
            np_array[i] = agx_list[i]
    elif agx_type == agx.Quat:
        np_array = np.zeros(shape=(4,), dtype=np.float64)
        for i in range(4):
            np_array[i] = agx_list[i]
    elif agx_type == agx.OrthoMatrix3x3:
        np_array = np.zeros(shape=(3, 3), dtype=np.float64)
        for i in range(3):
            row = agx_list.getRow(i)
            for j in range(3):
                np_array[i, j] = row[j]
    else:
        logger.warning('Conversion for {} type is not supported.'.format(agx_type))

    return np_array

def compute_segments_pos(sim):
    segments_pos = []
    dlo = agxCable.Cable.find(sim, "DLO")
    segment_iterator = dlo.begin()
    n_segments = dlo.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            pos = segment_iterator.getGeometry().getPosition()
            segments_pos.append(to_numpy_array(pos))
            segment_iterator.inc()

    return segments_pos