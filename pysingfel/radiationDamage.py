from .toolbox import *
from .FileIO import *
import os


def generateRotations(uniformRotation, rotationAxis, numQuaternions):
    """
    Return Quaternions saving the rotations to the particle.
    """

    if uniformRotation is None:
        # No rotation desired, init quaternions as (1,0,0,0)
        Quaternions =  np.empty((numQuaternions, 4))
        Quaternions[:,0] = 1.
        Quaternions[:,1:] = 0.

        return Quaternions

    # Case uniform:
    if uniformRotation:
        if rotationAxis == 'y' or rotationAxis == 'z':
            return pointsOn1Sphere(numQuaternions, rotationAxis)
        elif rotationAxis == 'xyz':
            return pointsOn4Sphere(numQuaternions)
    else:
        # Case non-uniform:
        Quaternions = np.zeros((numQuaternions, 4))
        for i in range(numQuaternions):
            Quaternions[i, :] = getRandomRotation(rotationAxis)
        return Quaternions


def rotateParticle(quaternion, particle):
    """
    Apply one quaternion to rotate the particle.
    """
    rot3D = quaternion2rot3D(quaternion)
    NewPos = np.dot(particle.atomPos, rot3D.T)
    particle.set_atomPos(NewPos)


def setEnergyFromFile(fname, beam):
    """
    Set photon energy from pmi file.
    """
    with h5py.File(fname, 'r') as f:
        if "photon_energy" in f['params'].keys():
            photon_energy = f.get('/params/photon_energy').value
        elif 'xparams' in f['params'].keys():
            lines = [l.split(' ') for l in f['params/xparams'].value.decode('utf-8').split("\n")]
            xparams = get_dict_from_lines(lines)
            photon_energy = xparams['EPH']
        else:
            # Legacy support: Try get from history.
            try:
                photon_energy = f.get('/history/parent/detail/params/photonEnergy').value
            except:
                raise

    beam.set_photon_energy(photon_energy)


def setFocusFromFile(fname, beam):
    """
    Set beam focus from pmi file.
    """
    with h5py.File(fname, 'r') as f:
        if "focus" in f['params'].keys():
            focus_xFWHM = f.get('/params/focus/xFWHM').value
            focus_yFWHM = f.get('/params/focus/yFWHM').value
        elif 'xparams' in f['params'].keys():
            lines = [l.split(' ') for l in f['params/xparams'].value.decode('utf-8').split("\n")]
            xparams = get_dict_from_lines(lines)
            diam = xparams['DIAM']
            focus_xFWHM = diam
            focus_yFWHM = diam

        else:
            try:
                focus_xFWHM = f.get('/history/parent/detail/misc/xFWHM').value
                focus_yFWHM = f.get('/history/parent/detail/misc/yFWHM').value
            except:
                raise

    beam.set_focus(focus_xFWHM, focus_yFWHM, shape='ellipse')


def setFluenceFromFile(fname, timeSlice, sliceInterval, beam):
    """
    Set beam fluence from pmi file.
    """
    n_phot = 0
    for i in range(sliceInterval):
        with h5py.File(fname, 'r') as f:
            datasetname = '/data/snp_' + '{0:07}'.format(timeSlice-i) + '/Nph'
            n_phot += f.get(datasetname).value
    beam.set_photonsPerPulse(n_phot)

def MakeOneDiffr(myQuaternions, counter, parameters, outputName):
    """
    Generate one diffraction pattern related to a certain rotation.
    Write results in output hdf5 file.
    """
    # Get parameters
    calculateCompton = parameters['calculateCompton']
    numDP = int(parameters['numDP'])
    numSlices = int(parameters['numSlices'])
    pmiStartID = int(parameters['pmiStartID'])
    pmiID = pmiStartID + counter / numDP
    sliceInterval = int(parameters['sliceInterval'])
    beamFile = parameters['beamFile']
    geomFile = parameters['geomFile']
    inputDir = parameters['inputDir']

    # Set up beam and detector from file
    det = Detector(geomFile)
    beam = Beam(beamFile)

    # If not given, read from pmi file later
    givenFluence = False
    if beam.get_photonsPerPulse() > 0:
        givenFluence = True
    givenPhotonEnergy = False
    if beam.get_photon_energy() > 0:
        givenPhotonEnergy = True
    givenFocusRadius = False
    if beam.get_focus() > 0:
        givenFocusRadius = True

    # Detector geometry
    px = det.get_numPix_x()
    py = det.get_numPix_x()

    # Setup quaternion.
    quaternion = myQuaternions[counter, :]

    # Input file
    inputName = os.path.join(inputDir, 'pmi_out_%07d.h5' % (pmiID) )

    # Set up diffraction geometry
    if not givenPhotonEnergy:
        setEnergyFromFile(inputName, beam)
    if not givenFocusRadius:
        setFocusFromFile(inputName, beam)
    det.init_dp(beam)

    done = False
    timeSlice = 0
    total_phot = 0
    detector_intensity = np.zeros((py, px))
    while not done:
        # set time slice to calculate diffraction pattern
        if timeSlice + sliceInterval >= numSlices:
            sliceInterval = numSlices - timeSlice
            done = True
        timeSlice += sliceInterval

        # load particle information
        datasetname = '/data/snp_%07d' % (timeSlice)
        particle = Particle(inputName, datasetname)
        rotateParticle(quaternion, particle)
        if not givenFluence:
            # sum up the photon fluence inside a sliceInterval
            setFluenceFromFile(inputName, timeSlice, sliceInterval, beam)
        total_phot += beam.get_photonsPerPulse()
        # Coherent contribution
        F_hkl_sq = calculate_molecularFormFactorSq(particle, det)
        # Incoherent contribution
        if calculateCompton:
            Compton = calculate_compton(particle, det)
        else:
            Compton = np.zeros((py, px))
        photon_field = F_hkl_sq + Compton
        detector_intensity += photon_field

    detector_intensity *= det.solidAngle * det.PolarCorr * beam.get_photonsPerPulsePerArea()


    detector_counts = convert_to_poisson(detector_intensity)
    saveAsDiffrOutFile(outputName, inputName, counter, detector_counts, detector_intensity, quaternion, det, beam)


def diffract(parameters):
    """
    Calculate all the diffraction patterns based on the parameters provided as a dictionary.
    Save all results in one single file. Not used in MPI.
    """
    pmiStartID = int(parameters['pmiStartID'])
    pmiEndID = int(parameters['pmiEndID'])
    numDP = int(parameters['numDP'])
    ntasks = (pmiEndID - pmiStartID + 1) * numDP
    rotationAxis = parameters['rotationAxis']
    uniformRotation = parameters['uniformRotation']
    myQuaternions = generateRotations(uniformRotation, rotationAxis, ntasks)
    outputName = parameters['outputDir'] + '/diffr_out_0000001.h5'
    if os.path.exists(outputName):
        os.remove(outputName)
    prepH5(outputName)
    for ntask in range(ntasks):
        MakeOneDiffr(myQuaternions, ntask, parameters, outputName)

def get_dict_from_lines(reader):
    """ Turn a list of [key, ' ', ..., value] elements into a dict.

    :params reader: An iterable that contains lists of strings in format [key, ' ', ' ', ..., value]
    :type: iterable (list, array, generator).

    """
    # These fields shall be handled as numeric data.
    numeric_keys = [
            'N',
            'Z',
            'DIST',
            'EPH',
            'NPH',
            'DIAM',
            'FLU_MAX',
            'T',
            'T0',
            'R0',
            'DT',
            'STEPS',
            'PROGRESS',
            'RANDSEED',
            'RSTARTE',
            ]
    # Initialize return dictionary.
    ret = dict()

    # Iteratoe through all lines.
    for line in reader:
        # Skip empty lines and comments.
        if line == [] or line == ['']:
            continue
        if line[0][0] == '#':
            continue

        # Get key-value pair (they're separated by random number of whitespaces.
        key, val = line[0], line[-1]

        # Fix numeric data.
        if key in numeric_keys:
            try:
                val = float(val)
            except:
                raise

        # Store on dict.
        ret[key] = val

    # Return finished dict.
    return ret


