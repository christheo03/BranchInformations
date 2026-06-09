import os
import re
import copy
import torch
import pandas as pd
import numpy as np
import random
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


SEED = 1
EMBED_DIM = 24
BATCH_SIZE = 512
EPOCHS = 150
LR = 0.001
PATIENCE = 30

CLASS_HNT = 0
CLASS_NB = 1
CLASS_HT = 2
N_CLASSES = 3

DATA_DIR = "../../results"

ALL_FILES = [
    "500.perlbench_r", "502.gcc_r", "505.mcf_r", "507.cactuBSSN_r",
    "508.namd_r", "510.parest_r", "511.povray_r", "519.lbm_r",
    "520.omnetpp_r","521.wrf_r", "523.xalancbmk_r", "525.x264_r", "526.blender_r",
    "527.cam4_r", "531.deepsjeng_r", "538.imagick_r", "541.leela_r",
    "544.nab_r", "548.exchange2_r", "549.fotonik3d_r", "554.roms_r",
    "557.xz_r", "503.bwaves_r",
]

TEST_FILES = [
    "503.bwaves_r",
    "520.omnetpp_r",
    "548.exchange2_r",
    "549.fotonik3d_r",
    "557.xz_r",
]

TRAIN_FILES = [f for f in ALL_FILES if f not in TEST_FILES]

REG_COLS = ["reg1", "reg2", "reg3", "wreg1", "wreg2"]

OPC_COLS = [
    "Opcode", "t_successor_ends", "f_successor_ends", "Flag_Instr_Opcode",
    "reg1_Op", "reg2_Op", "reg3_Op",
    "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
    "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5",
]

ROUT_COL = "Routine_Type"

ROUTINE_TYPE_MAP = {
    "NonLeaf": 1,
    "Leaf": 2,
    "Recursive": 3,
}

CONT_COLS = [
    "Offset",
]

BIN_COLS = [
    "br_is_loop_header",
    "t_dominates", "t_post_dominates", "t_is_loop_head",
    "t_is_backedge", "t_is_loop_exit", "t_has_call",
    "f_dominates", "f_post_dominates", "f_is_loop_head",
    "f_is_backedge", "f_is_loop_exit", "f_has_call",
    "Same_BBL", "taken_ubd", "fall_ubd", "taken_store", "fall_store",
]

DROP_COLUMNS = [
    "Taken", "Executed", "Regs_Read", "Regs_Write",
    "branch_bb_addr", "taken_bb_addr", "fall_bb_addr",
    "Address", "Flag_Write_PC","wreg3",
    "Size",
    "Prev_Size_1", "Prev_Size_2", "Prev_Size_3", "Prev_Size_4", "Prev_Size_5",
    "Next_Size_1", "Next_Size_2", "Next_Size_3", "Next_Size_4", "Next_Size_5",

]