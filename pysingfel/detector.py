from pysingfel.beam import *


class Detector(object):
    def __init__(self, fname):
        self.d = 0  # (m) detector distance
        self.pix_width = 0  # (m)
        self.pix_height = 0  # (m)
        self.px = 0  # number of pixels in x
        self.py = 0  # number of pixels in y
        self.numPix = 0  # total number of pixels (px*py)
        self.cx = 0  # center of detector in x
        self.cy = 0  # center of detector in y
        self.q_xyz = None  # would be defined in ini_dp function
        self.q_mod = None  # pixel reciprocal space
        self.pixSpace = None  # pixel reciprocal space in Angstrom
        self.pixSpaceMax = None  # max pixel reciprocal space in Angstrom
        self.solidAngle = None  # solid angle
        self.PolarCorr = None  # Polarization correction (+ thomson?)
        if fname is not None:
            self.readGeomFile(fname)

    # setters and getters
    def set_detector_dist(self, dist):
        self.d = dist

    def get_detector_dist(self):
        return self.d

    def set_pix_width(self, width):
        self.pix_width = width

    def get_pix_width(self):
        return self.pix_width

    def set_pix_height(self, height):
        self.pix_height = height

    def get_pix_height(self):
        return self.pix_height

    def set_numPix_x(self, x):
        self.px = x

    def get_numPix_x(self):
        return self.px

    def set_numPix_y(self, y):
        self.py = y

    def get_numPix_y(self):
        return self.py

    def set_numPix(self, y, x):
        self.py = y
        self.px = x
        self.numPix = x * y

    def set_center_x(self, x):
        self.cx = x

    def get_center_x(self):
        return self.cx

    def set_center_y(self, y):
        self.cy = y

    def get_center_y(self):
        return self.cy

    # Read from Geom file
    def readGeomFile(self, fname):
        with open(fname) as f:
            content = f.readlines()
            for line in content:
                if line[0] != '#' and line[0] != ';' and len(line) > 1 and '=' in line:
                    line = line.replace(' ','')
                    line = line.replace('\t','')
                    key, value = line.split('=')
                    identifier, key = key.split('/')
                    if key == 'd':
                        self.d = float(value)
                    if key == 'pix_width':
                        self.pix_width = float(value)
                        self.pix_height = self.pix_width
                    if key == 'px':
                        self.px = int(value)
                        self.py = self.px
                        self.numPix = self.px ** 2
                        self.cx = (self.px - 1) / 2.
                        self.cy = (self.py - 1) / 2.

    def init_dp(self, beam):
        """
        Initialize diffraction pattern based on beam object input.
        q_xyz: wavevector at each pixel of the detector
        q_mod: magnitude of the wavevectorat each pixel of the detector
        pixSpace: reshape 2D pixel space of q_xyz into 1D pixel space
        """
        # set beam object variables and assign q to each pixel
        q_xyz = np.zeros((self.py, self.px, 3))
        solidAngle = np.zeros((self.py, self.px))
        twoTheta = np.zeros((self.py, self.px))
        PolarCorr = np.zeros((self.py, self.px))
        theta = beam.Polarization
        PolarVec = np.array([np.cos(theta), np.sin(theta), 0])  # Polarization unit vector
        for ind_x in range(self.px):
            for ind_y in range(self.py):
                rx = (ind_x - self.cx) * self.pix_width  # equivalent to dividing by pixel resolution
                ry = (ind_y - self.cy) * self.pix_height
                r = np.sqrt(rx**2 + ry**2)
                pixDist = np.sqrt(r**2 + self.d**2)  # distance from interaction to pixel center in real space
                twoTheta[ind_y, ind_x] = np.arctan2(r, self.d)
                q_xyz[ind_y, ind_x, 0] = beam.get_wavenumber() * rx / pixDist
                q_xyz[ind_y, ind_x, 1] = beam.get_wavenumber() * ry / pixDist
                q_xyz[ind_y, ind_x, 2] = beam.get_wavenumber() * (self.d - pixDist) / pixDist
                ss = self.pix_width**2 / (4 * pixDist**2 + self.pix_width**2)
                solidAngle[ind_y, ind_x] = 4 * np.arcsin(ss)
                v = np.array([rx, ry, self.d])
                v = v / np.linalg.norm(v)
                PolarCorr[ind_y, ind_x] = 1 - np.dot(PolarVec, v)**2
        q_mod = np.sqrt(np.sum(q_xyz**2, axis=2))  # 2D matrix (magnitude of the q vectors)
        re = 2.81793870e-15  # classical electron radius (m)
        PolarCorr = re**2 * PolarCorr

        # Setup pixel space
        pixSpace = np.reshape(q_xyz, (self.numPix, 3),  order='F')  # order: column-wise
        pixSpace = pixSpace * 1e-10  # (A)
        pix_mod = np.sqrt(np.sum(pixSpace**2, axis=1))
        pixSpaceMax = max(pix_mod)
        inc_res = (self.px - 1) / (2. * pixSpaceMax / np.sqrt(2))
        pixSpace = pixSpace * inc_res
        pix_mod = np.sqrt(np.sum(pixSpace ** 2, axis=1))
        # pixSpaceMax = self.cx FIXME: something fishy going on here
        self.q_xyz = q_xyz
        self.q_mod = q_mod
        self.pixSpace = pixSpace
        self.pixSpaceMax = pixSpaceMax
        self.solidAngle = solidAngle
        self.PolarCorr = PolarCorr
