import numpy as np
import matplotlib.pyplot as plt
from Renderer import Renderer
from Visualization import Visualization

class EKFSLAM(object):
    """A class for implementing EKF-based SLAM

        Attributes
        ----------
        mu :           The mean vector (numpy.array)
        Sigma :        The covariance matrix (numpy.array)
        R :            The process model covariance matrix (numpy.array)
        Q :            The measurement model covariance matrix (numpy.array)
        XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
        MGT :          Ground-truth map (optional, may be None)

        Methods
        -------
        prediction :   Perform the prediction step
        update :       Perform the measurement update step
        augmentState : Add a new landmark(s) to the state
        run :          Main EKF-SLAM loop
        render :       Render the filter
    """

    def __init__(self, mu, Sigma, R, Q, XGT = None, MGT = None):
        """Initialize the class

            Args
            ----------
            mu :           The initial mean vector (numpy.array)
            Sigma :        The initial covariance matrix (numpy.array)
            R :            The process model covariance matrix (numpy.array)
            Q :            The measurement model covariance matrix (numpy.array)
            XGT :          Array of ground-truth poses (optional, may be None) (numpy.array)
            MGT :          Ground-truth map (optional, may be None)
        """
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        self.XGT = XGT
        self.MGT = MGT

        if (self.XGT is not None and self.MGT is not None):
            xmin = min(np.amin(XGT[1, :]) - 2, np.amin(MGT[1, :]) - 2)
            xmax = min(np.amax(XGT[1, :]) + 2, np.amax(MGT[1, :]) + 2)
            ymin = min(np.amin(XGT[2, :]) - 2, np.amin(MGT[2, :]) - 2)
            ymax = min(np.amax(XGT[2, :]) + 2, np.amax(MGT[2, :]) + 2)
            xLim = np.array((xmin, xmax))
            yLim = np.array((ymin, ymax))
        else:
            xLim = np.array((-8.0, 8.0))
            yLim = np.array((-8.0, 8.0))

        self.renderer = Renderer(xLim, yLim, 3, 'red', 'green')

        # Draws the ground-truth map
        if self.MGT is not None:
            self.renderer.drawMap(self.MGT)


        # You may find it useful to keep a dictionary that maps a feature ID
        # to the corresponding index in the mean vector and covariance matrix
        self.mapLUT = {}

    def prediction(self, u):
        d, dtheta = u
        x, y, theta = self.mu
        
        return self.mu, self.sigma

    def update(self, z, id):
        """Perform the measurement update step to compute the posterior
           belief given the predictive posterior (mean and covariance) and
           the measurement data

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark (int)
        """
        z1 = z[0]
        z2 = z[1]
        x = self.mu[0]
        y = self.mu[1]
        theta = self.mu[2]
        W = np.random.multivariate_normal(0, cov = self.Q)
        w1 = W[0] 
        w2 = W[1] 

        Kt = self.Sigma @ self.H.T @ np.linalg.inv(self.H @ self.Sigma @ self.H.T + self.Q)
        
        z_hat = np.array([x+z1*np.cos(theta) - z2*np.sin(theta)+w1, y+z1*np.sin(theta)+z2*np.cos(theta+w2) ])
        innovation = z - z_hat

        self.mu = self.mu + np.dot(Kt, innovation)
        self.Sigma = (np.eye(3)-Kt@self.H)@self.Sigma

    def augmentState(self, z, id):
        """Augment the state vector to include the new landmark

            Args
            ----------
            z :  The Cartesian coordinates of the landmark
                 in the robot's reference frame (numpy.array)
            id : The ID of the observed landmark
        """
        x = self.mu[0]
        y = self.mu[1]
        theta = self.mu[2]

        # extract landmark position in robot's reference frame
        px = z[0]
        py = z[1]

        # compute the position of the landmark in the world frame
        lx = x + px * np.cos(theta) - py * np.sin(theta)
        ly = y + px * np.sin(theta) + py * np.cos(theta)

        # augment the state vector with the new landmark
        self.mu = np.append(self.mu, [lx, ly])

        # update the landmark Jacobian matrix
        H = np.array([[-np.cos(theta), -np.sin(theta), (lx-x)* -np.sin(theta) + (ly -y) * np.cos(theta), np.cos(theta), np.sin(theta)],
                    [np.sin(theta), -np.cos(theta), (lx-x)* -1* np.cos(theta)+(ly-y)*-np.sin(theta), -np.sin(theta), np.cos(theta)]])
        
        # augment the covariance matrix with the new landmark
        

        
        

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def run(self, U, Z):
        """The main loop of EKF-based SLAM

            Args
            ----------
            U :   Array of control inputs, one column per time step (numpy.array)
            Z :   Array of landmark observations in which each column
                  [t; id; x; y] denotes a separate measurement and is
                  represented by the time step (t), feature id (id),
                  and the observed (x, y) position relative to the robot
        """
        # TODO: Your code goes here
        pass

        # You may want to call the visualization function between filter steps where
        #       self.XGT[1:4, t] is the column of XGT containing the pose the current iteration
        #       Zt are the columns in Z for the current iteration
        #       self.mapLUT is a dictionary where the landmark IDs are the keys
        #                   and the index in mu is the value
        #
        # self.renderer.render(self.mu, self.Sigma, self.XGT[1:4, t], Zt, self.mapLUT)
