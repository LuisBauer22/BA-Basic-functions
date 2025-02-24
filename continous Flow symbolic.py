# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:56:40 2025

@author: luis-
"""

#Cont. flow symbolic


import sympy as smp

#reproduce the last script with symbols
#we can use this to check derivatives and divergence

theta_s, theta1_s, theta0_s, x1_s, x2_s, X_s, u1_s, u2_s, u_s, u_grad_s, t_s = smp.symbols('theta_s theta1_s theta0_s x1_s x2_s X_s u1_s u2_s u_s u_grad_s t_s')



theta_s = smp.Matrix([theta0_s, theta1_s])

u_s = smp.Matrix([-theta1_s *x1_s+0.1*x2_s, 
                  theta0_s**2 - 3*x1_s +theta0_s *x2_s])

u_grad_s =smp.Matrix([smp.diff(u_s[0], x1_s),
                      smp.diff(u_s[1], x2_s)])

u_div_s = smp.diff(u_s[0], x1_s) + smp.diff(u_s[1], x2_s)

