r"""
Usage of the scapy and numpy library is partially referenced in the code below.
https://github.com/munhouiani/Deep-Packet/blob/master/preprocessing.py
"""

from __future__ import annotations

import pathlib
import sys

if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from typing import TYPE_CHECKING, Optional

from scapy.packet import Padding, raw
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.dns import DNS

from numpy import uint8, float32, ndarray, frombuffer, divide, pad

from utils.flags import TCPFlag as Tflg

if TYPE_CHECKING:
    from scapy.packet import Packet

    pass


class PacketPreprocessor:
    def __init__(self, packet_len: int = 1500) -> None:
        self.__packet_len = packet_len
        pass

    def process(self, packet: Packet) -> Optional[ndarray]:
        if self.__check_discard(packet):
            return None

        packet = self.__mask_ip(packet)
        packet = self.__remove_l2_header(packet)
        packet = self.__pad_l4_header(packet)
        divided_packet = self.__divide_bytes(packet)
        padded_packet = self.__pad_packet(divided_packet)
        return padded_packet

    def __remove_l2_header(self, packet: Packet) -> Packet:
        if Ether in packet:
            packet = packet[Ether].payload
            pass
        return packet

    def __pad_l4_header(self, packet: Packet) -> Packet:
        if UDP in packet:
            udp_payload = packet[UDP].payload.copy()
            padding = Padding()
            padding.load = "\x00" * 12
            ip_udp_header = packet.copy()
            ip_udp_header[UDP].remove_payload()
            packet = ip_udp_header / padding / udp_payload
            pass
        return packet

    def __check_discard(self, packet: Packet) -> bool:
        if self.__check_tcp_handshake(packet) or self.__check_dns(packet):
            return True
        return False

    def __check_tcp_handshake(self, packet: Packet) -> bool:
        if TCP in packet and (packet.flags & (Tflg.SYN | Tflg.ACK | Tflg.FIN)):
            tcp_layers = packet[TCP].payload.layers()
            if not tcp_layers or (Padding in tcp_layers and len(tcp_layers) == 1):
                return True
            pass
        return False

    def __check_dns(self, packet: Packet) -> bool:
        if DNS in packet:
            return True
        return False

    def __mask_ip(self, packet: Packet) -> Packet:
        if IP in packet:
            packet[IP].src = "0.0.0.0"
            packet[IP].dst = "0.0.0.0"
            pass
        return packet

    def __divide_bytes(self, packet: Packet) -> ndarray:
        return divide(
            frombuffer(raw(packet), dtype=uint8)[0 : self.__packet_len],
            255,
            dtype=float32,
        )

    def __pad_packet(self, half_arr: ndarray) -> ndarray:
        pad_len = self.__packet_len - len(half_arr)
        if pad_len > 0:
            return pad(half_arr, pad_width=(0, pad_len), constant_values=0)
        return half_arr

    pass
