# settings.py

# Runner
MPI = 1
PRIME1 = 20
RUN_STEPS = 250000
WRITE_INTERVAL = 5000

# Provider
PROVIDER = True
NUM_PROVIDERS = 38
PROVIDER_MAXQ = 1
PROVIDER_ALPHA = 0.0375
PROVIDER_DELTA = 0.025
Q_PROVIDE = 0.5

# Taker
TAKER = True
NUM_TAKERS = 50
TAKER_MAXQ = 1
TAKER_MU = 0.001

# Informed
INFORMED = False
INFORMED_MAXQ = 1
INFORMED_RUN_LENGTH = 1
INFORMED_MU = 0.005

# Penny Jumper
PENNYJUMPER = False
PJ_ALPHA = 0.01

# Market Maker
MARKETMAKER = True
NUM_MMS = 1
ARR_INT = 1
MM_MAXQ = 5
GENETIC_INT = 250

# Q-Take
Q_TAKE = True
WHITENOISE = 0.001
C_LAMBDA = 10.0
LAMBDA0 = 100

# Order Imbalance
OI_SIGNAL = [-16, -8, -6, -4, -2, 0, 0, 2, 4, 6, 8, 16, -8, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 8]
OI_HIST_LEN = 5
OI_NUM_CHROMS = 100
OI_COND_LEN = len(OI_SIGNAL)
OI_ACTION_LEN = 6 # 6 bits -> lhs bit is +1/-1 and 2^5 - 1 = 31 for a range of -31 -> +31 (symm==True)
OI_COND_PROBS = [0.05, 0.05, 0.9]
OI_ACTION_MUTATE_P = 0.06
OI_COND_CROSS_P = 0.1
OI_COND_MUTATE_P = 0.06
OI_THETA = 0.02
OI_KEEP_PCT = 0.8
OI_SYMM = True
OI_WEIGHTS = False

# Order Flow
OF_SIGNAL = [0, 1, 2, 4, 8, 16, 32, 64, 0, 1, 2, 3, 4, 6, 8, 12]
OF_HIST_LEN = 5
OF_NUM_CHROMS = 100
OF_COND_LEN = len(OF_SIGNAL)
OF_ACTION_LEN = 5 # 5 bits -> 2^5 - 1 = 31 for a range of 0 -> +31 (symm==False)
OF_COND_PROBS = [0.05, 0.05, 0.9]
OF_ACTION_MUTATE_P = 0.06
OF_COND_CROSS_P = 0.1
OF_COND_MUTATE_P = 0.06
OF_THETA = 0.02
OF_KEEP_PCT = 0.8
OF_SYMM = False
OF_WEIGHTS = False
