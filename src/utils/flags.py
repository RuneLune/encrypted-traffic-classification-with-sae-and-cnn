import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))


class TCPFlag:
    r"""TCP Flags."""

    FIN = 0b00000001
    SYN = 0b00000010
    RST = 0b00000100
    PSH = 0b00001000
    ACK = 0b00010000
    URG = 0b00100000
    ECE = 0b01000000
    CWR = 0b10000000

    # def __add__(self, other):
    #     return self | other

    # def __truediv__(self, other):
    #     return self | other

    # def __or__(self, other):
    #     return self | other

    pass
