import h5py

from pysingfel.diffraction import *


def prepH5(outputName):
    """
    Create output file, prepare top level groups, write metadata.
    """
    with h5py.File(outputName, 'w') as f:
        # Generate top level groups
        f.create_group('data')
        f.create_group('params')
        f.create_group('misc')
        f.create_group('info')

        # Write metadata
        # Package format version
        f.create_dataset('info/package_version', data=np.string_('SingFEL v0.2.0'))
        # Contact
        f.create_dataset('info/contact', data=np.string_('Carsten Fortmann-Grote <carsten.grote@xfel.eu>'))
        # Data Description
        f.create_dataset('info/data_description', data=np.string_('This dataset contains diffraction patterns generated using SingFEL.'))
        # Method Description
        f.create_dataset('info/method_description', data=np.string_('Form factors of the radiation damaged molecules are calculated in time slices. At each time slice, the coherent scattering is calculated and incoherently added to the final diffraction pattern (/data/nnnnnnn/diffr). Finally, Poissonian noise is added to the diffraction pattern (/data/nnnnnnn/data).'))
        # Data format version
        f.create_dataset('version', data=np.string_('0.2'))


def saveAsDiffrOutFile(outputName, inputName, counter, detector_counts, detector_intensity, quaternion, det, beam):
    """
    Save simulation results as new dataset in to the h5py file prepared before.
    """
    with h5py.File(outputName, 'a') as f:
        group_name = '/data/' + '{0:07}'.format(counter + 1) + '/'
        f.create_dataset(group_name + 'data', data=detector_counts)
        f.create_dataset(group_name + 'diffr', data=detector_intensity)
        f.create_dataset(group_name + 'angle', data=quaternion)

        # Link history from input pmi file into output diffr file
        if inputName is not None and inputName.split('.')[-1] == '.h5':
            group_name_history = group_name + 'history/parent/detail/'
            f[group_name_history + 'data'] = h5py.ExternalLink(inputName, 'data')
            f[group_name_history + 'info'] = h5py.ExternalLink(inputName, 'info')
            f[group_name_history + 'misc'] = h5py.ExternalLink(inputName, 'misc')
            f[group_name_history + 'params'] = h5py.ExternalLink(inputName, 'params')
            f[group_name_history + 'version'] = h5py.ExternalLink(inputName, 'version')
            f[group_name + '/history/parent/parent'] = h5py.ExternalLink(inputName, 'history/parent')

        # Parameters
        if 'geom' not in list(f['params'].keys()) and 'beam' not in list(f['params'].keys()):
            # Geometry
            f.create_dataset('params/geom/detectorDist', data=det.get_detector_dist())
            f.create_dataset('params/geom/pixelWidth', data=det.get_pix_width())
            f.create_dataset('params/geom/pixelHeight', data=det.get_pix_height())
            f.create_dataset('params/geom/mask', data=np.ones((det.py, det.px)))

            # Photons
            f.create_dataset('params/beam/photonEnergy', data=beam.get_photon_energy())
            f.create_dataset('params/beam/focusArea', data=beam.get_focus_area())







