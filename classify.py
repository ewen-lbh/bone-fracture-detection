from enum import Enum

Point = tuple[int, int]
Segment = tuple[Point, Point, float]

class FractureType(str, Enum):
    TRANSVERSALE = "transversale"
    OBLIQUE = "oblique"
    LONGITUDINALE = "longitudinale"
    AILE_PAPILLON = "aile_papillon"
    COMMINUTIVE = "comminutive"

def orthogonal(angle: float, ε: float = 2.0) -> bool:
    return angle - 90 <= ε

def classify(segments: list[Segment]) ->  FractureType:
    largest_angle = max(angle for _, _, angle in segments) 

    if orthogonal(largest_angle):
        return FractureType.TRANSVERSALE
    return FractureType.OBLIQUE
