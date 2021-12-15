'''
    :project:   Kinematic based backstepping controller implementation for 
                unicycle model mobile robots.
    
    :authors:   manyetar@gmail.com
                github:saltin0

    :reference: will be added.

    Notice:     You can take initial gains by (5.0,5.0,10.0) for tuning Kx,Ky,Kr
                parameters. There is no analytic way to tune this parameter.
                You can use PSO (Particle Swarm Optimization), GA (Genetic Algorithm)
                or etc. 
'''

import numpy as np
from math import sqrt

class BackStepping():
    def __init__(self,Kx,Ky,Kr,no_back_move=False):
        '''
            :function:  init-Initialize the necessary parameters.

            :param:     Kx-x position error coefficient
            :param:     Ky-y position error coefficient
            :param:     Kr-heading angle error coefficient
            :param:     no_back_move- If velocity can not below zero is intended
                        this parameter must be True.
        '''
        self.Kx = Kx
        self.Ky = Ky
        self.Kr = Kr

        # Desired velocity init params 
        self.v_des = 0.0
        
        self.no_back_move = no_back_move


    def set_setpoint(self,x_des,y_des,psi_des,x_des_dot,y_des_dot,r_des = 0.0):
        '''
            :function:  set_setpoint-Give the controller reference values.
        '''

        self.x_des = x_des
        self.y_des = y_des
        self.psi_des = psi_des

        self.v_des = sqrt(x_des_dot**2+y_des_dot**2)
        self.r_des = r_des

    def execute(self,x,y,psi):
        '''
            :function:  execute-Kinematic based backstepping controller.
                        controller rule will be explained.
            
            :params:    x-x Position of the robot
            :params:    y-y Position of the robot
            :params:    psi-Heading angle of the robot

            :return:    (ctrl_v,ctrl_w)-Differential drive robot linear velocity
                        and angular velocity reference values.

        '''
        # Error in global frame
        self.err_matrix = np.array(([self.x_des-x,self.y_des -y , self.psi_des - psi]),dtype=np.float32).reshape(3,1)
        # Error in body frame
        self.err_body = np.dot(self.rotation_matrix(psi),self.err_matrix)
        ex = self.err_body[0,0]
        ey = self.err_body[1,0]
        e_psi = self.err_body[2,0]
        # Differential drive robot linear velocity and angular velocity references
        ''' 
            controller rule:
                v = backstepping 
        '''
        ctrl_v = self.v_des*np.cos(e_psi) + self.Kx*ex
        ctrl_w = self.r_des + self.Ky*self.v_des*ey + self.Kr*self.v_des*np.sin(e_psi)
        # Because of imu no backward move allowed
        if self.no_back_move == True:
            if ctrl_v <= 0:
                ctrl_v = 0

        # Return debug params for debugging
        self.ctrl_v = ctrl_v
        self.ctrl_w = ctrl_w


        return (ctrl_v,ctrl_w)

    def rotation_matrix(self,psi):
        '''
            :function:  rotation_matrix-It takes the heading angle and calculates the 
                        necessary rotation matrix to work with body reference frame.

            :return:    rot_matrix
        '''
        rot_matrix = np.array(([[np.cos(psi),np.sin(psi),0.0],[-np.sin(psi),np.cos(psi),0.0],[0.0,0.0,1.0]]),dtype=np.float32).reshape(3,3)
        return rot_matrix

    @property
    def debugger(self):
        '''
            :property:  debugger-It takes all info in the controller 
                        and return in place where controller is used.

            :return:    debug_vals
        '''
        debug_vals = {
            "err_body" : self.err_body,
            "controller_output" : (self.ctrl_v,self.ctrl_w)
        }
        print(debug_vals)
        return debug_vals


