"""
@author: Alejandro Gonzalez, Ivana Collado, Sebastian
        Perez

Environment of an Unmanned Surface Vehicle with an
Adaptive Sliding Mode Controller to train collision
avoidance on the OpenAI Gym library.
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class UsvAsmcCaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        #Integral step (or derivative) for 100 Hz
        self.integral_step = 0.01

        #USV model coefficients
        self.X_u_dot = -2.25
        self.Y_v_dot = -23.13
        self.Y_r_dot = -1.31
        self.N_v_dot = -16.41
        self.N_r_dot = -2.79
        self.Xuu = 0
        self.Yvv = -99.99
        self.Yvr = -5.49
        self.Yrv = -5.49
        self.Yrr = -8.8
        self.Nvv = -5.49
        self.Nvr = -8.8
        self.Nrv = -8.8
        self.Nrr = -3.49
        self.m = 30
        self.Iz = 4.1
        self.B = 0.41
        self.c = 0.78

        #ASMC gains
        self.k_u = 0.1
        self.k_psi = 0.2
        self.kmin_u = 0.05
        self.kmin_psi = 0.2
        self.k2_u = 0.02
        self.k2_psi = 0.1
        self.mu_u = 0.05
        self.mu_psi = 0.1
        self.lambda_u = 0.001
        self.lambda_psi = 1

        #Second order filter gains (for r_d)
        self.f1 = 2.
        self.f2 = 2.
        self.f3 = 2.

        #Overall vector variables
        self.state = None
        self.position = None
        self.aux_vars = None
        self.last = None
        self.target = None
        self.so_filter = None

        #Obstacle variables
        self.num_obs = None
        self.posx = None #array
        self.posy = None #array
        self.radius = None #array
        
        # Sensor vector column 0 = senor angle column 1 = distance mesured
        self.sensors = np.zeros((800, 2))
        self.sensor_span = (2/3)*(2*np.pi)
        self.lidar_resolution = 0.00524 #angle resolution in radians
        self.sector_size = 32 # number of points per sector
        self.sector_num = 25 # number of sectors

        # Boat radius
        self.boat_radius = 0.5
        self.safety_radius = 0.3
        self.safety_distance = 0.1

        #Map limits in meters
        self.max_y = 10
        self.min_y = -10
        self.max_x = 30
        self.min_x = -10

        #Variable for the visualizer
        self.viewer = None

        #Min and max actions 
        # velocity 
        self.min_action0 = 0
        self.max_action0 = 1.4
        # angle (change to -pi and pi if necessary)
        self.min_action1 = -np.pi/2
        self.max_action1 = np.pi/2

        #Reward associated functions anf gains
        self.w_y = 0.4
        self.w_u = 0.2
        self.w_chi = 0.4
        self.k_ye = 0.5
        self.k_uu = 15.0
        self.gamma_theta = 1.0 
        self.gamma_x = 1.0
        self.epsilon = 1.0
        self.sigma_ye = 1.
        self.lambda_reward = 0.9
        self.w_action0 = 0.2
        self.w_action1 = 0.2
        self.c_action0 = 1. / np.power((self.max_action0/2-self.min_action0/2)/self.integral_step, 2)
        self.c_action1 = 1. / np.power((self.max_action1/2-self.min_action1/2)/self.integral_step, 2)

        #Min and max values of the state
        self.min_u = -1.5
        self.max_u = 1.5
        self.min_v = -1.0
        self.max_v = 1.0
        self.min_r = -1.
        self.max_r = 1.
        self.min_ye = -10.
        self.max_ye = 10.
        self.min_ye_dot = -1.5
        self.max_ye_dot = 1.5
        self.min_chi_ak = -np.pi
        self.max_chi_ak = np.pi
        self.min_u_ref = 0.3
        self.max_u_ref = 1.4
        self.min_sectors = np.zeros((25))
        self.max_sectors = np.full((25), 100.)
        self.sectors = np.zeros((25))

        #Min and max state vectors 
        self.low_state = np.hstack((self.min_u, self.min_v, self.min_r, self.min_ye, self.min_ye_dot, self.min_chi_ak, self.min_u_ref, self.min_sectors, self.min_action0, self.min_action1))
        self.high_state = np.hstack((self.max_u, self.max_v, self.max_r, self.max_ye, self.max_ye_dot, self.max_chi_ak, self.max_u_ref, self.max_sectors, self.max_action0, self.max_action1))

        self.min_action = np.array([self.min_action0, self.min_action1])
        self.max_action = np.array([self.max_action0, self.max_action1])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

    def step(self, action):
        '''
        @name: step
        @brief: ASMC and USV step, add obstacles and sensors.
        @param: action: vector of actions
        @return: state: state vector
                 reward: reward from the current action
                 done: if finished
        '''
        #Read overall vector variables
        state = self.state
        position = self.position
        aux_vars = self.aux_vars
        last = self.last
        target = self.target
        so_filter = self.so_filter

        #Change from vectors to scalars
        u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last = state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7:32], state[32], state[33]
        x, y, psi = position
        e_u_int, Ka_u, Ka_psi = aux_vars
        x_dot_last, y_dot_last, psi_dot_last, u_dot_last, v_dot_last, r_dot_last, e_u_last, Ka_dot_u_last, Ka_dot_psi_last = last
        x_0, y_0, u_ref, ak, x_d, y_d = target
        psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot = so_filter

        #Create model related vectors
        eta = np.array([x, y, psi])
        upsilon = np.array([u, v, r])
        eta_dot_last = np.array([x_dot_last, y_dot_last, psi_dot_last])
        upsilon_dot_last = np.array([u_dot_last, v_dot_last, r_dot_last])

        #Calculate action derivative for reward
        action_dot0 = (action[0] - action0_last)/self.integral_step
        action_dot1 = (action[1] - action1_last)/self.integral_step
        action_last0 = action[0]
        action_last1 = action[1]

        beta = np.math.asin(upsilon[0]/(0.001 + np.sqrt(upsilon[0]*upsilon[0]+upsilon[1]*upsilon[1])))
        chi = psi + beta
        chi = np.where(np.greater(np.abs(chi), np.pi), (np.sign(chi))*(np.abs(chi)-2*np.pi), chi)

        #Compute the desired heading
        psi_d = chi + action[1]
        psi_d = np.where(np.greater(np.abs(psi_d), np.pi), (np.sign(psi_d))*(np.abs(psi_d)-2*np.pi), psi_d)

        #Second order filter to compute desired yaw rate
        r_d = (psi_d - psi_d_last) / self.integral_step
        psi_d_last = psi_d
        o_dot_dot = (((r_d - o_last) * self.f1) - (self.f3 * o_dot_last)) * self.f2
        o_dot = (self.integral_step)*(o_dot_dot + o_dot_dot_last)/2 + o_dot
        o = (self.integral_step)*(o_dot + o_dot_last)/2 + o
        r_d = o
        o_last = o
        o_dot_last = o_dot
        o_dot_dot_last = o_dot_dot

        #Compute variable hydrodynamic coefficients
        Xu = -25
        Xuu = 0
        if(abs(upsilon[0]) > 1.2):
            Xu = 64.55
            Xuu = -70.92

        Yv = 0.5*(-40*1000*abs(upsilon[1])) * \
            (1.1+0.0045*(1.01/0.09) - 0.1*(0.27/0.09)+0.016*(np.power((0.27/0.09), 2)))
        Yr = 6*(-3.141592*1000) * \
            np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01
        Nv = 0.06*(-3.141592*1000) * \
            np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01
        Nr = 0.02*(-3.141592*1000) * \
            np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01*1.01

        #Rewrite USV model in simplified components f and g
        g_u = 1 / (self.m - self.X_u_dot)
        g_psi = 1 / (self.Iz - self.N_r_dot)
        f_u = (((self.m - self.Y_v_dot)*upsilon[1]*upsilon[2] + (Xuu*np.abs(upsilon[0]) + Xu*upsilon[0])) / (self.m - self.X_u_dot))
        f_psi = (((-self.X_u_dot + self.Y_v_dot)*upsilon[0]*upsilon[1] + (Nr * upsilon[2])) / (self.Iz - self.N_r_dot))

        #Compute heading error
        e_psi = psi_d - eta[2]
        e_psi = np.where(np.greater(np.abs(e_psi), np.pi), (np.sign(e_psi))*(np.abs(e_psi)-2*np.pi), e_psi)
        e_psi_dot = r_d - upsilon[2]
        #bs_e_psi = np.abs(e_psi)

        #Compute desired speed (unnecessary if DNN gives it)
        u_d = action[0]

        #Compute speed error
        e_u = u_d - upsilon[0]
        e_u_int = self.integral_step*(e_u + e_u_last)/2 + e_u_int

        #Create sliding surfaces for speed and heading
        sigma_u = e_u + self.lambda_u * e_u_int
        sigma_psi = e_psi_dot + self.lambda_psi * e_psi

        #Compute ASMC gain derivatives
        Ka_dot_u = np.where(np.greater(Ka_u, self.kmin_u), self.k_u * np.sign(np.abs(sigma_u) - self.mu_u), self.kmin_u)
        Ka_dot_psi = np.where(np.greater(Ka_psi, self.kmin_psi), self.k_psi * np.sign(np.abs(sigma_psi) - self.mu_psi), self.kmin_psi)

        #Compute gains
        Ka_u = self.integral_step*(Ka_dot_u + Ka_dot_u_last)/2 + Ka_u
        Ka_dot_u_last = Ka_dot_u

        Ka_psi = self.integral_step*(Ka_dot_psi + Ka_dot_psi_last)/2 + Ka_psi
        Ka_dot_psi_last = Ka_dot_psi

        #Compute ASMC for speed and heading
        ua_u = (-Ka_u * np.power(np.abs(sigma_u), 0.5) * np.sign(sigma_u)) - (self.k2_u * sigma_u)
        ua_psi = (-Ka_psi * np.power(np.abs(sigma_psi), 0.5) * np.sign(sigma_psi)) - (self.k2_psi * sigma_psi)

        #Compute control inputs for speed and heading
        Tx = ((self.lambda_u * e_u) - f_u - ua_u) / g_u
        Tz = ((self.lambda_psi * e_psi) - f_psi - ua_psi) / g_psi

        #Compute both thrusters and saturate their values
        Tport = (Tx / 2) + (Tz / self.B)
        Tstbd = (Tx / (2*self.c)) - (Tz / (self.B*self.c))

        Tport = np.where(np.greater(Tport, 36.5), 36.5, Tport)
        Tport = np.where(np.less(Tport, -30), -30, Tport)
        Tstbd = np.where(np.greater(Tstbd, 36.5), 36.5, Tstbd)
        Tstbd = np.where(np.less(Tstbd, -30), -30, Tstbd)

        #Compute USV model matrices
        M = np.array([[self.m - self.X_u_dot, 0, 0],
                      [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                      [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])

        T = np.array([Tport + self.c*Tstbd, 0, 0.5*self.B*(Tport - self.c*Tstbd)])

        CRB = np.array([[0, 0, 0 - self.m * upsilon[1]],
                        [0, 0, self.m * upsilon[0]],
                        [self.m * upsilon[1], 0 - self.m * upsilon[0], 0]])
        
        CA = np.array([[0, 0, 2 * ((self.Y_v_dot*upsilon[1]) + ((self.Y_r_dot + self.N_v_dot)/2) * upsilon[2])],
                       [0, 0, 0 - self.X_u_dot * self.m * upsilon[0]],
                       [2*(((0 - self.Y_v_dot) * upsilon[1]) - ((self.Y_r_dot+self.N_v_dot)/2) * upsilon[2]), self.X_u_dot * self.m * upsilon[0], 0]])

        C = CRB + CA

        Dl = np.array([[0 - Xu, 0, 0],
                       [0, 0 - Yv, 0 - Yr],
                       [0, 0 - Nv, 0 - Nr]])

        Dn = np.array([[Xuu * abs(upsilon[0]), 0, 0],
                       [0, self.Yvv * abs(upsilon[1]) + self.Yvr * abs(upsilon[2]), self.Yrv *
                        abs(upsilon[1]) + self.Yrr * abs(upsilon[2])],
                       [0, self.Nvv * abs(upsilon[1]) + self.Nvr * abs(upsilon[2]), self.Nrv * abs(upsilon[1]) + self.Nrr * abs(upsilon[2])]])

        D = Dl - Dn

        #Compute acceleration and velocity in body
        upsilon_dot = np.matmul(np.linalg.inv(
            M), (T - np.matmul(C, upsilon) - np.matmul(D, upsilon)))
        upsilon = (self.integral_step) * (upsilon_dot +
                                               upsilon_dot_last)/2 + upsilon  # integral
        upsilon_dot_last = upsilon_dot

        #Rotation matrix
        J = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0],
                      [np.sin(eta[2]), np.cos(eta[2]), 0],
                      [0, 0, 1]])

        #Compute NED position
        eta_dot = np.matmul(J, upsilon)  # transformation into local reference frame
        eta = (self.integral_step)*(eta_dot+eta_dot_last)/2 + eta  # integral
        eta_dot_last = eta_dot

        psi = eta[2]
        psi = np.where(np.greater(np.abs(psi), np.pi), (np.sign(psi))*(np.abs(psi)-2*np.pi), psi)

        beta = np.math.asin(upsilon[0]/(0.001 + np.sqrt(upsilon[0]*upsilon[0]+upsilon[1]*upsilon[1])))
        chi = psi + beta
        chi = np.where(np.greater(np.abs(chi), np.pi), (np.sign(chi))*(np.abs(chi)-2*np.pi), chi)
        #Compute angle between USV and path
        chi_ak = chi - ak
        chi_ak = np.where(np.greater(np.abs(chi_ak), np.pi), (np.sign(chi_ak))*(np.abs(chi_ak)-2*np.pi), chi_ak)
        psi_ak = psi - ak
        psi_ak = np.where(np.greater(np.abs(psi_ak), np.pi), (np.sign(psi_ak))*(np.abs(psi_ak)-2*np.pi), psi_ak)


        #Compute cross-track error
        ye = -(eta[0] - x_0)*np.math.sin(ak) + (eta[1] - y_0)*np.math.cos(ak)
        ye_abs = np.abs(ye)

        collision = False
        # Compute collision
        distance = np.empty([self.num_obs])
        for i in range(self.num_obs):
            distance[i] = np.sqrt(np.power((self.posx[i] - eta[0]),2) + np.power((self.posy[i]-eta[1]),2))
            distance[i] = distance[i] - self.radius[i] - self.boat_radius - self.safety_radius 
            if distance[i] < self.safety_distance:
                collision == True

        # Compute sensor readings
        obs_order = np.argsort(distance) # order obstacles in closest to furthest
        for i in range(len(self.sensors)):
            self.sensors[i][0]= -np.pi*2/3 + i*self.lidar_resolution
            m = np.math.tan(self.sensors[i][0])
            for j in range(self.num_obs):
                obs_index = obs_order[j]
                posx,posy = self.ned_to_body(self.posx[obs_index], self.posy[obs_index], eta[0], eta[1], eta[2])
                delta = ((self.radius[j]*self.radius[j])*(1+(m*m))) - (posx-m*posy)*(posx-m*posy)
                if delta >= 0: # intersection
                    y1 = (posy+posx*m+np.sqrt(delta))/(1+m*m)
                    y2 = (posy+posx*m-np.sqrt(delta))/(1+m*m)
                    x1 = (posy*m+posx*m*m+m*np.sqrt(delta))/(1+m*m)
                    x2 = (posy*m+posx*m*m-m*np.sqrt(delta))/(1+m*m)
                    distance1 = np.sqrt(x1*x1+y1*y1)
                    distance2 = np.sqrt(x2*x2+y2*y2)
                    if distance1 > distance2:
                      self.sensors[i][1] = distance2
                    else:
                      self.sensors[i][1] = distance1
                    break
                else:
                    self.sensors[i][1] = 100

        # Feasability pooling: compute sectors
        sectors = self.max_sectors
        for i in range(self.sector_num): # loop through sectors
            x = self.sensors[i*self.sector_size:(i+1)*self.sector_size,1]
            x_ordered = np.argsort(x)
            for j in range(self.sector_size): # loop through 
                x_index = x_ordered[j]
                arc_length = self.lidar_resolution*x[x_index]
                opening_width = arc_length/2
                opening_found = False
                for k in range(self.sector_size):
                    if x[k] > x[x_index]:
                        opening_width = opening_width + arc_length
                        if opening_width > (self.boat_radius+self.safety_radius):
                            opening_found = True
                            break
                    else:
                        opening_width = opening_width + arc_length/2
                        if opening_width > (self.boat_radius+self.safety_radius):
                            opening_found = True
                            break
                        opening_width = 0
                if opening_found == False:
                    sectors[i] = x[x_index]
        self.sectors = sectors
        sectors = (1-sectors/100)            

        #Compute reward 
        reward = self.compute_reward(ye_abs, chi_ak, action_dot0, action_dot1, collision, u_ref, u, v)

        #Compute velocities relative to path (for ye derivative as ye_dot = v_ak)
        xe_dot, ye_dot = self.body_to_path(upsilon[0], upsilon[1], psi_ak)

        #If ye is too large or USV went backwards too much, abort
        if collision:
            done = True
        else:
            done = False

        #Fill overall vector variables 
        self.state = np.hstack((upsilon[0], upsilon[1], upsilon[2], ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last))
        self.position = np.array([eta[0], eta[1], psi])
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])
        self.last = np.array([eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1], upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])
        self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])

        #Reshape state
        state = self.state.reshape(self.observation_space.shape[0])

        return state, reward, done, {}

    def reset(self):

        x = np.random.uniform(low=-2.5, high=2.5)
        y = np.random.uniform(low=-5.0, high=5.0)
        psi = np.random.uniform(low=-np.pi, high=np.pi)
        eta = np.array([x, y])
        upsilon = np.array([0.,0.,0.])
        eta_dot_last = np.array([0.,0.,0.])
        upsilon_dot_last = np.array([0.,0.,0.])
        action0_last = 0.0
        action1_last = psi
        e_u_int = 0.
        Ka_u = 0.
        Ka_psi = 0.
        e_u_last = 0.
        Ka_dot_u_last = 0.
        Ka_dot_psi_last = 0.
        psi_d_last = psi
        o_dot_dot_last = 0.
        o_dot_last = 0.
        o_last = 0.
        o_dot_dot = 0.
        o_dot = 0.
        o = 0.
        # Start and Final position
        x_0 = np.random.uniform(low=-2.5, high=2.5)
        y_0 = np.random.uniform(low=-5.0, high=5.0)
        x_d = np.random.uniform(low=15, high=30)
        y_d = y_0
        # Desired speed
        u_ref = np.random.uniform(low=self.min_u_ref, high=self.max_u_ref)
        # number of obstacles 
        self.num_obs = np.random.random_integers(low=20, high=40)
        # array of positions in x and y and radius
        self.posx = np.random.normal(15,10,size=(self.num_obs,1))
        self.posy = np.random.uniform(low=-10, high=10, size=(self.num_obs,1))
        self.radius = np.random.uniform(low=0.1, high=1.5, size=(self.num_obs,1))

        ak = np.math.atan2(y_d-y_0,x_d-x_0)
        ak = np.float32(ak)

        psi_ak = psi - ak
        psi_ak = np.where(np.greater(np.abs(psi_ak), np.pi), np.sign(psi_ak)*(np.abs(psi_ak)-2*np.pi), psi_ak)
        psi_ak = np.float32(psi_ak)
        ye = -(x - x_0)*np.math.sin(ak) + (y - y_0)*np.math.cos(ak)

        xe_dot, ye_dot = self.body_to_path(upsilon[0], upsilon[1], psi_ak)

        collision = False
        # Compute collision
        distance = np.empty([self.num_obs])
        for i in range(self.num_obs):
            distance[i] = np.sqrt(np.power((self.posx[i] - eta[0]),2) + np.power((self.posy[i]-eta[1]),2))
            distance[i] = distance[i] - self.radius[i] - self.boat_radius - self.safety_radius 
            if distance[i] < self.safety_distance:
                collision == True

        # Compute sensor readings
        obs_order = np.argsort(distance) # order obstacles in closest to furthest
        for i in range(len(self.sensors)):
            self.sensors[i][0]= -np.pi*2/3 + i*self.lidar_resolution
            m = np.math.tan(self.sensors[i][0])
            for j in range(self.num_obs):
                obs_index = obs_order[j]
                posx,posy = self.ned_to_body(self.posx[obs_index], self.posy[obs_index], eta[0], eta[1], psi)
                delta = ((self.radius[j]*self.radius[j])*(1+(m*m))) - (posx-m*posy)*(posx-m*posy)
                if delta >= 0: # intersection
                    y1 = (posy+posx*m+np.sqrt(delta))/(1+m*m)
                    y2 = (posy+posx*m-np.sqrt(delta))/(1+m*m)
                    x1 = (posy*m+posx*m*m+m*np.sqrt(delta))/(1+m*m)
                    x2 = (posy*m+posx*m*m-m*np.sqrt(delta))/(1+m*m)
                    distance1 = np.sqrt(x1*x1+y1*y1)
                    distance2 = np.sqrt(x2*x2+y2*y2)
                    if distance1 > distance2:
                      self.sensors[i][1] = distance2
                    else:
                      self.sensors[i][1] = distance1
                    break
                else:
                    self.sensors[i][1] = 100

        # Feasability pooling: compute sectors
        sectors = self.max_sectors
        for i in range(self.sector_num): # loop through sectors
            x = self.sensors[i*self.sector_size:(i+1)*self.sector_size,1]
            x_ordered = np.argsort(x)
            for j in range(self.sector_size): # loop through 
                x_index = x_ordered[j]
                arc_length = self.lidar_resolution*x[x_index]
                opening_width = arc_length/2
                opening_found = False
                for k in range(self.sector_size):
                    if x[k] > x[x_index]:
                        opening_width = opening_width + arc_length
                        if opening_width > (self.boat_radius+self.safety_radius):
                            opening_found = True
                            break
                    else:
                        opening_width = opening_width + arc_length/2
                        if opening_width > (self.boat_radius+self.safety_radius):
                            opening_found = True
                            break
                        opening_width = 0
                if opening_found == False:
                    sectors[i] = x[x_index]
        self.sectors = sectors
        sectors = (1-sectors/100)

        self.state = np.hstack((upsilon[0], upsilon[1], upsilon[2], ye, ye_dot, psi_ak, u_ref, sectors, action0_last, action1_last))
        self.position = np.array([eta[0], eta[1], psi])
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])
        self.last = np.array([eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1], upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])
        self.target = np.array([x_0, y_0, u_ref, ak, x_d, y_d])
        self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])

        state = self.state.reshape(self.observation_space.shape[0])

        return state

    def render(self, mode='human'):

        screen_width = 400
        screen_height = 800

        world_width = self.max_y - self.min_y
        scale = screen_width/world_width
        boat_width = 15
        boat_height = 20

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            clearance = -10
            l, r, t, b, c, m = -boat_width/2, boat_width/2, boat_height, 0, 0, boat_height/2
            boat = rendering.FilledPolygon([(l,b), (l,m), (c,t), (r,m), (r,b)])
            boat.add_attr(rendering.Transform(translation=(0, clearance)))
            self.boat_trans = rendering.Transform()
            boat.add_attr(self.boat_trans)
            self.viewer.add_geom(boat)

        x_0 = (self.min_x - self.min_x)*scale
        y_0 = (self.target[1] - self.min_y)*scale
        x_d = (self.max_x - self.min_x)*scale
        y_d = (self.target[5] - self.min_y)*scale
        start = (y_0, x_0)
        end = (y_d, x_d)

        self.viewer.draw_line(start, end)

        for i in range(self.num_obs):
            transform2 = rendering.Transform(translation=((self.posy[i]-self.min_y)*scale, (self.posx[i]-self.min_x)*scale))  # Relative offset
            self.viewer.draw_circle(self.radius[i]*scale, 30, True, color=(0, 0, 255)).add_attr(transform2)

        x = self.position[0]
        y = self.position[1]
        psi = self.position[2]

        safety = rendering.Transform(translation=((y-self.min_y)*scale, (x-self.min_x)*scale))  # Relative offset
        self.viewer.draw_circle((self.boat_radius+self.safety_radius)*scale, 30, False, color=(255, 0, 0)).add_attr(safety)
        self.boat_trans.set_translation((y-self.min_y)*scale, (x-self.min_x)*scale)
        self.boat_trans.set_rotation(-psi)

        self.viewer.draw_line(start, end)

        angle = -(2/3)*np.pi + psi + 0.08377
        angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle)*(np.abs(angle)-2*np.pi), angle)
        for i in range(self.sector_num):
          initial = ((y-self.min_y)*scale, (x-self.min_x)*scale)
          m = np.math.tan(angle)
          x_f = self.sectors[i]*np.math.cos(angle) + x - self.min_x
          y_f = self.sectors[i]*np.math.sin(angle) + y - self.min_y
          final = (y_f*scale, x_f*scale)
          self.viewer.draw_line(initial, final)
          angle = angle + .1675
          angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle)*(np.abs(angle)-2*np.pi), angle)


        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):

        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def compute_reward(self, ye, chi_ak, action_dot0, action_dot1, collision, u_ref, u, v):
        if (collision == False):
            chi_ak = np.abs(chi_ak)
            # Cross tracking reward
            reward_ye = np.where(np.greater(ye, self.sigma_ye), np.exp(-self.k_ye*ye), np.exp(-self.k_ye*np.power(ye, 2)/self.sigma_ye))
            # Velocity reward
            reward_u = np.exp(-self.k_uu*np.abs(u_ref-np.sqrt(u*u+v*v)))
            # Angle reward
            reward_chi = np.cos(chi_ak)
            # Action velocity gradual change reward
            reward_a0 = np.math.tanh(-self.c_action0*np.power(action_dot0, 2))
            # Action angle gradual change reward
            reward_a1 = np.math.tanh(-self.c_action1*np.power(action_dot1, 2))
            # Path following reward 
            reward_pf = self.w_y*reward_ye + self.w_chi*reward_chi + self.w_u*reward_u + self.w_action0*reward_a0 + self.w_action1*reward_a1 
            # Obstacle avoidance reward
            numerator = 0.0
            denominator = 0.0
            for i in range(len(self.sensors)):
                numerator = numerator + (1./(1.+self.gamma_theta*self.sensors[i][1]))*(1./(self.gamma_x*np.max([self.sensors[i][0], self.epsilon])))
                denominator = denominator + 1/(1+np.abs(self.gamma_theta*self.sensors[i][1]))
            reward_oa = -numerator/denominator
            # Total non-collision reward
            reward = self.lambda_reward*reward_pf + (1-self.lambda_reward)*reward_oa
        else:
            # Collision Reward
            reward = -1
        return reward

    def body_to_path(self, x2, y2, alpha):
        '''
        @name: body_to_path
        @brief: Coordinate transformation between body and path reference frames.
        @param: x2: target x coordinate in body reference frame
                y2: target y coordinate in body reference frame
        @return: path_x2: target x coordinate in path reference frame
                 path_y2: target y coordinate in path reference frame
        '''
        p = np.array([x2, y2])
        J = self.rotation_matrix(alpha)
        n = J.dot(p)
        path_x2 = n[0]
        path_y2 = n[1]
        return (path_x2, path_y2)

    def rotation_matrix(self, angle):
        '''
        @name: rotation_matrix
        @brief: Transformation matrix template.
        @param: angle: angle of rotation
        @return: J: transformation matrix
        '''
        J = np.array([[np.math.cos(angle), -1*np.math.sin(angle)],
                      [np.math.sin(angle), np.math.cos(angle)]])
        return (J)

    def ned_to_body(self, ned_x2, ned_y2, ned_xboat, ned_yboat, psi):
        '''
        @name: ned_to_ned
        @brief: Coordinate transformation between NED and body reference frames.
        @param: ned_x2: target x coordinate in ned reference frame
                ned_y2: target y coordinate in ned reference frame
                ned_xboat: robot x regarding NED
                ned_yboat: robot y regarding NED
                psi: robot angle regarding NED
        @return: body_x2: target x coordinate in body reference frame
                body_y2: target y coordinate in body reference frame
        '''
        n = np.array([ned_x2 - ned_xboat, ned_y2 - ned_yboat])
        J = self.rotation_matrix(psi)
        J = np.linalg.inv(J)
        b = J.dot(n)
        body_x2 = b[0]
        body_y2 = b[1]
        return (body_x2, body_y2)
