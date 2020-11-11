import numpy as np


def corrCoeff(X, Y):
    x = X.reshape(-1)
    y = Y.reshape(-1)
    x -= np.mean(x)
    y -= np.mean(y)
    return np.dot(x, y) / np.sqrt(np.dot(x, x)*np.dot(y, y))


# Converters between different descriptions of 3D rotation.
def angleAxis2rot3D(axis, theta):
    """
    Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.
    """
    if len(axis) is not 3:
        raise ValueError('Number of axis element must be 3!')
    axis = axis.astype(float)
    axis /= np.linalg.norm(axis)
    a = axis[0]
    b = axis[1]
    c = axis[2]
    cosTheta = np.cos(theta)
    bracket = 1 - cosTheta
    aBracket = a * bracket
    bBracket = b * bracket
    cBracket = c * bracket
    sinTheta = np.sin(theta)
    aSinTheta = a * sinTheta
    bSinTheta = b * sinTheta
    cSinTheta = c * sinTheta
    rot3D = np.array([[a*aBracket+cosTheta, a*bBracket-cSinTheta, a*cBracket+bSinTheta],
                     [b*aBracket+cSinTheta, b*bBracket+cosTheta, b*cBracket-aSinTheta],
                     [c*aBracket-bSinTheta, c*bBracket+aSinTheta, c*cBracket+cosTheta]])
    return rot3D


def euler2rot3D(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to a rotation matrix in 3D.
    """
    Rphi = np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    Rtheta = np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]])
    Rpsi = np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    return np.dot(Rpsi, np.dot(Rtheta, Rphi))


def euler2quaternion(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to quaternion description.
    """
    if abs(psi) == 0 and abs(theta) == 0 and abs(phi) == 0:
        quaternion = np.array([1., 0., 0., 0.])
    else:
        R = euler2rot3D(psi, theta, phi)
        W = np.array([R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0]])
        if W[0] >= 0:
            W /= np.linalg.norm(W)
        else:
            W /= np.linalg.norm(W) * -1
        theta = np.arccos(0.5 * (np.trace(R) - 1))
        CCisTheta = corrCoeff(R, angleAxis2rot3D(W, theta))
        CCisNegTheta = corrCoeff(R, angleAxis2rot3D(W, -theta))
        if CCisNegTheta > CCisTheta:
            theta = -theta
        quaternion = np.array([np.cos(theta/2.), np.sin(theta/2.)*W[0], np.sin(theta/2.)*W[1], np.sin(theta/2.)*W[2]])
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def quaternion2AngleAxis(quaternion):
    """
    Convert quaternion to a right hand rotation theta about certain axis.
    """
    HA = np.arccos(quaternion[0])
    theta = 2 * HA
    if theta < np.finfo(float).eps:
        theta = 0
        axis = np.array([1, 0, 0])
    else:
        axis = quaternion[[1, 2, 3]] / np.sin(HA)
    return theta, axis


def quaternion2rot3D(quaternion):
    """
    Convert quaternion to a rotation matrix in 3D.
    Use zyz convention after Heymann (2005)
    """
    theta, axis = quaternion2AngleAxis(quaternion)
    return angleAxis2rot3D(axis, theta)


# Functions to generate rotations for different cases: uniform(1d), uniform(3d), random.
def pointsOn1Sphere(numPts, rotationAxis):
    """
    Given number of points and axis of rotation, distribute evenly on the surface of a 1-sphere (circle).
    """
    points = np.zeros((numPts, 4))
    incAng = 360. / numPts
    myAng = 0
    if rotationAxis == 'y':
        for i in range(numPts):
            points[i, :] = euler2quaternion(0, myAng * np.pi / 180, 0)
            myAng += incAng
    elif rotationAxis == 'z':
        for i in range(numPts):
            points[i, :] = euler2quaternion(0, 0, myAng * np.pi / 180)
            myAng += incAng
    return points


def pointsOn4Sphere(numPts):
    """
    Given number of points, distribute evenly on hyper surface of a 4-sphere.
    """
    points = np.zeros((2*numPts, 4))
    N = 4
    surfaceArea = N * np.pi ** (N/2) / (N/2)  # for even N
    delta = np.exp(np.log(surfaceArea / numPts) / 3)
    Iter = 0
    ind = 0
    maxIter = 1000
    while ind != numPts and Iter < maxIter:
        ind = 0
        deltaW1 = delta
        w1 = 0.5 * deltaW1
        while w1 < np.pi:
            q0 = np.cos(w1)
            deltaW2 = deltaW1 / np.sin(w1)
            w2 = 0.5 * deltaW2
            while w2 < np.pi:
                q1 = np.sin(w1) * np.cos(w2)
                deltaW3 = deltaW2 / np.sin(w2)
                w3 = 0.5 * deltaW3
                while w3 < 2 * np.pi:
                    q2 = np.sin(w1) * np.sin(w2) * np.cos(w3)
                    q3 = np.sin(w1) * np.sin(w2) * np.sin(w3)
                    points[ind, :] = np.array([q0, q1, q2, q3])
                    ind += 1
                    w3 += deltaW3
                w2 += deltaW2
            w1 += deltaW1
        delta *= np.exp(np.log(float(ind) / numPts) / 3)
        Iter += 1
    return points[0:numPts, :]


def getRandomRotation(rotationAxis):
    """
    Generate random rotation.
    """
    if rotationAxis == 'y':
        u = np.random.random() * 2 * np.pi  # random angle between [0, 2pi]
        return euler2quaternion(0, u, 0)
    else:
        u = np.random.rand(3)  # uniform random distribution in the [0,1] interval
        # generate uniform random quaternion on SO(3)
        return np.array([np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]), np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
                         np.sqrt(u[0]) * np.sin(2*np.pi*u[2]), np.sqrt(u[0]) * np.cos(2*np.pi*u[2])])


def convert_to_poisson(dp):
    """
    Add poisson noise to a certain diffraction pattern dp.
    """
    return np.random.poisson(dp)
