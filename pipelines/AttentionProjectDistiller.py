from utils import DISTILLER
from utils import build_function, build_model

import torch
import torch.nn as nn

@DISTILLER.register()
class AttentionProjectDistiller():
    def __init__(self,
                 device,
                 teacher,
                 teacher_init_weight,
                 ta1,
                 ta2,
                 ta3,
                 label_loss_function,
                 loss_function_f1,
                 loss_function_f2,
                 loss_function_f3,
                 ALPHA,
                 BETA1,
                 BETA2,
                 BETA3,
                 GAMMA,
                 ta4=None,
                 loss_function_f4=None,
                 BETA4=None):
        self.teacher = build_model(teacher).to(device)
        self.teacher.load_state_dict(torch.load(teacher_init_weight))
        self.teacher.eval()
        self.ta1 = build_model(ta1).to(device)
        self.ta2 = build_model(ta2).to(device)
        self.ta3 = build_model(ta3).to(device)
        self.label_loss_function = build_function(label_loss_function)
        self.loss_function_f1 = build_function(loss_function_f1)
        self.loss_function_f2 = build_function(loss_function_f2)
        self.loss_function_f3 = build_function(loss_function_f3)
        self.BETA1 = BETA1
        self.BETA2 = BETA2
        self.BETA3 = BETA3
        self.GAMMA = GAMMA
        self.s4 = False
        if ta4 is not None:
            self.ta4 = build_model(ta4).to(device)
            self.loss_function_f4 = build_function(loss_function_f4)
            self.BETA4 = BETA4
            self.s4 = True
    
    def distill(self, data, labels, student):
        s_pred = student(data)
        t_pred = self.teacher(data)
        node1 = self.ta1(s_pred[0])
        node2 = self.ta2(s_pred[1])
        node3 = self.ta3(s_pred[2])
        node1_loss = self.logger.log(self.loss_function_f1(node1, t_pred[0]), "loss_t12")
        node2_loss = self.logger.log(self.loss_function_f2(node2, t_pred[1]), "loss_t23")
        node3_loss = self.logger.log(self.loss_function_f3(node3, t_pred[2]), "loss_t34")
        if self.s4:
            node4 = self.ta4(s_pred[3])
            node4_loss = self.logger.log(self.loss_function_f3(node4, t_pred[3]), "loss_t44")
        label_loss = self.logger.log(self.label_loss_function(s_pred[-1], t_pred[-1]), "loss_labelt")
        if self.s4:
            return node1_loss * self.BETA1 + node2_loss * self.BETA2 + node3_loss * self.BETA3 + node4_loss * self.BETA4 + label_loss * self.GAMMA
        return node1_loss * self.BETA1 + node2_loss * self.BETA2 + node3_loss * self.BETA3 + label_loss * self.GAMMA

    def init_logger(self, logger):
        self.logger = logger

"""
                 interL, intraL,
                 inter1, intra1,
                 inter2, intra2,
                 inter3, intra3,
                 Temperature_L,
                 Temperature_1,
                 Temperature_2,
                 Temperature_3,
"""