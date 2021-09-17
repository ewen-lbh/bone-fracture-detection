"""
Corrects tilted images
"""
from numpy import pi

from classify import Segment
from utils import mean

tau = 2 * pi


def image_tilt(
    segments: list[Segment], ε: float = 2 / 180 * tau, max_tilt: float = 15 / 180 * tau
) -> float:
    """
    Returns the angle (in radians) the image is tilted at

    Relies on the fact the the majority of detected segments will be (physically) upright

    Angles greater than max_tilt will not be included in the calculation.
    """

    mean_angle = mean(angle for _, _, angle in segments if angle <= max_tilt)
    print(f"[tilt] mean angle is {mean_angle/tau*180}°")
    print(
        f"[tilt] (ε={ε/tau*180}°) kept angles are [{' '.join('%d° <%2f°>' % ((a/tau*180), abs(a - mean_angle)/tau*180) for _, _, a in segments if not (abs(a - mean_angle) <= ε))}]"
    )
    try:
        return mean(angle for _, _, angle in segments if abs(angle - mean_angle) <= ε)
    except ZeroDivisionError:
        return 0
