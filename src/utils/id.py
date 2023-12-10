import re

from typing import List, Optional

from regex_spm import search_in


class AppID:
    r"""Application ID class"""

    AIM_CHAT = 0
    EMAIL = 1
    FACEBOOK = 2
    FTPS = 3
    GMAIL = 4
    HANGOUTS = 5
    ICQ = 6
    NETFLIX = 7
    SCP = 8
    SFTP = 9
    SKYPE = 10
    SPOTIFY = 11
    TORRENT = 12
    TOR = 13
    VOIPBUSTER = 14
    VIMEO = 15
    YOUTUBE = 16
    pass


class TrafficID:
    r"""Traffic ID class"""

    CHAT = 0
    EMAIL = 1
    FILE_TRANSFER = 2
    STREAMING = 3
    TORRENT = 4
    VOIP = 5
    VPN_CHAT = 6
    VPN_FILE_TRANSFER = 7
    VPN_EMAIL = 8
    VPN_STREAMING = 9
    VPN_TORRENT = 10
    VPN_VOIP = 11
    pass


AppName: List[str] = [
    "AIM chat",
    "Email",
    "Facebook",
    "FTPS",
    "Gmail",
    "Hangouts",
    "ICQ",
    "Netflix",
    "SCP",
    "SFTP",
    "Skype",
    "Spotify",
    "Torrent",
    "Tor",
    "VoipBuster",
    "Vimeo",
    "YouTube",
]
r"""Application name list"""

TrafficName: List[str] = [
    "Chat",
    "Email",
    "File transfer",
    "Streaming",
    "Torrent",
    "VoIP",
    "VPN: Chat",
    "VPN: File transfer",
    "VPN: Email",
    "VPN: Streaming",
    "VPN: Torrent",
    "VPN: VoIP",
]
r"""Traffic name list"""


class AppRegex:
    r"""
    Class for storing regular expressions for application names
    """

    AIM_CHAT = re.compile(r"^(vpn_)?aim_?chat", re.IGNORECASE)
    EMAIL = re.compile(r"^(vpn_)?email", re.IGNORECASE)
    FACEBOOK = re.compile(r"^(vpn_)?facebook", re.IGNORECASE)
    FTPS = re.compile(r"^(vpn_)?ftps", re.IGNORECASE)
    GMAIL = re.compile(r"^(vpn_)?gmail", re.IGNORECASE)
    HANGOUTS = re.compile(r"^(vpn_)?hangout", re.IGNORECASE)
    ICQ = re.compile(r"^(vpn_)?icq", re.IGNORECASE)
    NETFLIX = re.compile(r"^(vpn_)?netflix", re.IGNORECASE)
    SCP = re.compile(r"^(vpn_)?scp", re.IGNORECASE)
    SFTP = re.compile(r"^(vpn_)?sftp", re.IGNORECASE)
    SKYPE = re.compile(r"^(vpn_)?skype", re.IGNORECASE)
    SPOTIFY = re.compile(r"^(vpn_)?spotify", re.IGNORECASE)
    TORRENT = re.compile(r"^(vpn_)?.*(torrent|vuze)", re.IGNORECASE)
    TOR = re.compile(r"^(vpn_)?tor", re.IGNORECASE)
    VOIPBUSTER = re.compile(r"^(vpn_)?voipbuster", re.IGNORECASE)
    VIMEO = re.compile(r"^(vpn_)?vimeo", re.IGNORECASE)
    YOUTUBE = re.compile(r"^(vpn_)?youtube", re.IGNORECASE)

    # def get_by_id(self, id: int) -> re.Pattern:
    #     match id:
    #         case AppID.AIM_CHAT:
    #             return self.AIM_CHAT
    #         case AppID.EMAIL:
    #             return self.EMAIL
    #         case AppID.FACEBOOK:
    #             return self.FACEBOOK
    #         case AppID.FTPS:
    #             return self.FTPS
    #         case AppID.GMAIL:
    #             return self.GMAIL
    #         case AppID.HANGOUTS:
    #             return self.HANGOUTS
    #         case AppID.ICQ:
    #             return self.ICQ
    #         case AppID.NETFLIX:
    #             return self.NETFLIX
    #         case AppID.SCP:
    #             return self.SCP
    #         case AppID.SFTP:
    #             return self.SFTP
    #         case AppID.SKYPE:
    #             return self.SKYPE
    #         case AppID.SPOTIFY:
    #             return self.SPOTIFY
    #         case AppID.TORRENT:
    #             return self.TORRENT
    #         case AppID.TOR:
    #             return self.TOR
    #         case AppID.VOIPBUSTER:
    #             return self.VOIPBUSTER
    #         case AppID.VIMEO:
    #             return self.VIMEO
    #         case AppID.YOUTUBE:
    #             return self.YOUTUBE
    #     return None

    pass


class TrafficRegex:
    r"""
    Class for storing regular expressions for traffic names
    """

    CHAT = re.compile(r"^(?!vpn_).*chat", re.IGNORECASE)
    EMAIL = re.compile(r"^(?!vpn_).*email", re.IGNORECASE)
    FILE_TRANSFER = re.compile(r"^(?!vpn_).*(ftp|scp|file)", re.IGNORECASE)
    STREAMING = re.compile(
        r"^(?!vpn_).*(video|netflix|spotify|vimeo|youtube)", re.IGNORECASE
    )
    TORRENT = re.compile(r"^(?!vpn_).*(torrent|vuze)", re.IGNORECASE)
    VOIP = re.compile(r"^(?!vpn_).*(audio|voipbuster)", re.IGNORECASE)
    VPN_CHAT = re.compile(r"^vpn_.*chat", re.IGNORECASE)
    VPN_FILE_TRANSFER = re.compile(r"^vpn_.*(ftp|scp|file)", re.IGNORECASE)
    VPN_EMAIL = re.compile(r"^vpn_.*email", re.IGNORECASE)
    VPN_STREAMING = re.compile(
        r"^vpn_.*(video|netflix|spotify|vimeo|youtube)", re.IGNORECASE
    )
    VPN_TORRENT = re.compile(r"^vpn_.*torrent", re.IGNORECASE)
    VPN_VOIP = re.compile(r"^vpn_.*(audio|voipbuster)", re.IGNORECASE)
    pass


def get_app_regex(app_id: int) -> Optional[re.Pattern]:
    r"""
    function for returning regular expression by application ID

    ### Args:
        `app_id` (int): application ID

    ### Returns:
        `app_regex` (Pattern): regular expression corresponding to application ID
    """

    match app_id:
        case AppID.AIM_CHAT:
            return AppRegex.AIM_CHAT
        case AppID.EMAIL:
            return AppRegex.EMAIL
        case AppID.FACEBOOK:
            return AppRegex.FACEBOOK
        case AppID.FTPS:
            return AppRegex.FTPS
        case AppID.GMAIL:
            return AppRegex.GMAIL
        case AppID.HANGOUTS:
            return AppRegex.HANGOUTS
        case AppID.ICQ:
            return AppRegex.ICQ
        case AppID.NETFLIX:
            return AppRegex.NETFLIX
        case AppID.SCP:
            return AppRegex.SCP
        case AppID.SFTP:
            return AppRegex.SFTP
        case AppID.SKYPE:
            return AppRegex.SKYPE
        case AppID.SPOTIFY:
            return AppRegex.SPOTIFY
        case AppID.TORRENT:
            return AppRegex.TORRENT
        case AppID.TOR:
            return AppRegex.TOR
        case AppID.VOIPBUSTER:
            return AppRegex.VOIPBUSTER
        case AppID.VIMEO:
            return AppRegex.VIMEO
        case AppID.YOUTUBE:
            return AppRegex.YOUTUBE

    return None


def get_traffic_regex(traffic_id: int) -> Optional[re.Pattern]:
    r"""
    function for returning regular expression by traffic ID

    ### Args:
        `traffic_id` (int): traffic ID

    ### Returns:
        `traffic_regex` (Pattern): regular expression corresponding to traffic ID
    """

    match traffic_id:
        case TrafficID.CHAT:
            return TrafficRegex.CHAT
        case TrafficID.EMAIL:
            return TrafficRegex.EMAIL
        case TrafficID.FILE_TRANSFER:
            return TrafficRegex.FILE_TRANSFER
        case TrafficID.STREAMING:
            return TrafficRegex.STREAMING
        case TrafficID.TORRENT:
            return TrafficRegex.TORRENT
        case TrafficID.VOIP:
            return TrafficRegex.VOIP
        case TrafficID.VPN_CHAT:
            return TrafficRegex.VPN_CHAT
        case TrafficID.VPN_FILE_TRANSFER:
            return TrafficRegex.VPN_FILE_TRANSFER
        case TrafficID.VPN_EMAIL:
            return TrafficRegex.VPN_EMAIL
        case TrafficID.VPN_STREAMING:
            return TrafficRegex.VPN_STREAMING
        case TrafficID.VPN_TORRENT:
            return TrafficRegex.VPN_TORRENT
        case TrafficID.VPN_VOIP:
            return TrafficRegex.VPN_VOIP

    return None


def get_app_id(filename: str) -> int:
    r"""
    function for extracting application ID from `filename` using regular expression

    ### Args:
        `filename` (str): name of PCAP file

    ### Returns:
        `app_id` (int): application ID, `-1` if invalid
    """

    match search_in(filename):
        case AppRegex.AIM_CHAT:
            return AppID.AIM_CHAT
        case AppRegex.EMAIL:
            return AppID.EMAIL
        case AppRegex.FACEBOOK:
            return AppID.FACEBOOK
        case AppRegex.FTPS:
            return AppID.FTPS
        case AppRegex.GMAIL:
            return AppID.GMAIL
        case AppRegex.HANGOUTS:
            return AppID.HANGOUTS
        case AppRegex.ICQ:
            return AppID.ICQ
        case AppRegex.NETFLIX:
            return AppID.NETFLIX
        case AppRegex.SCP:
            return AppID.SCP
        case AppRegex.SFTP:
            return AppID.SFTP
        case AppRegex.SKYPE:
            return AppID.SKYPE
        case AppRegex.SPOTIFY:
            return AppID.SPOTIFY
        case AppRegex.TORRENT:
            return AppID.TORRENT
        case AppRegex.TOR:
            return AppID.TOR
        case AppRegex.VOIPBUSTER:
            return AppID.VOIPBUSTER
        case AppRegex.VIMEO:
            return AppID.VIMEO
        case AppRegex.YOUTUBE:
            return AppID.YOUTUBE

    return -1


def get_traffic_id(filename: str) -> int:
    r"""
    function for extracting traffic ID from `filename` using regular expression

    ### Args:
        `filename` (str): name of PCAP file

    ### Returns:
        `traffic_id` (int): traffic ID, `-1` if invalid
    """

    match search_in(filename):
        case TrafficRegex.CHAT:
            return TrafficID.CHAT
        case TrafficRegex.EMAIL:
            return TrafficID.EMAIL
        case TrafficRegex.FILE_TRANSFER:
            return TrafficID.FILE_TRANSFER
        case TrafficRegex.STREAMING:
            return TrafficID.STREAMING
        case TrafficRegex.TORRENT:
            return TrafficID.TORRENT
        case TrafficRegex.VOIP:
            return TrafficID.VOIP
        case TrafficRegex.VPN_CHAT:
            return TrafficID.VPN_CHAT
        case TrafficRegex.VPN_FILE_TRANSFER:
            return TrafficID.VPN_FILE_TRANSFER
        case TrafficRegex.VPN_EMAIL:
            return TrafficID.VPN_EMAIL
        case TrafficRegex.VPN_STREAMING:
            return TrafficID.VPN_STREAMING
        case TrafficRegex.VPN_TORRENT:
            return TrafficID.VPN_TORRENT
        case TrafficRegex.VPN_VOIP:
            return TrafficID.VPN_VOIP

    return -1


if __name__ == "__main__":
    pass
