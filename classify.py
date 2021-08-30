from enum import Enum
from typing import NamedTuple

Point = tuple[int, int]
Segment = tuple[Point, Point, float]


class Scale(int, Enum):
    """
    A scale to be used to classify fractures and give a treatment suggestion.

    """

    @property
    def stable(self) -> bool:
        raise NotImplementedError()

    @property
    def treatment_suggestion(self) -> set[str]:
        raise NotImplementedError()

    @classmethod
    def compute(cls, fracture: "Fracture") -> "Scale":
        raise NotImplementedError()


class Patient(NamedTuple):
    age: float  # in years

    @property
    def en_croissance(self) -> bool:
        # TODO
        return self.age <= 18


LONG_BONES_NAMES = {"umérus", "fémur"}


class Fracture(NamedTuple):
    _bones: set[str]
    _segments: list[Segment]
    patient: Patient

    @property
    def bones(self) -> set[str]:
        return {b.lower().strip() for b in self._bones}

    @property
    def largest_angle(self) -> float:
        return max(angle for _, _, angle in self._segments)

    @property
    def applicable_scales(self) -> set[Scale]:
        scales = set()
        if "hip" in self.bones:
            scales.add(Garden)
        if self.patient.en_croissance and self.bones & LONG_BONES_NAMES:
            scales.add(Salter)

        return scales


class FractureType(str, Enum):
    TRANSVERSALE = "transversale"
    OBLIQUE = "oblique"
    LONGITUDINALE = "longitudinale"
    AILE_PAPILLON = "aile_papillon"
    COMMINUTIVE = "comminutive"

    @classmethod
    def compute(cls, fracture: Fracture) -> "FractureType":
        # TODO wip
        if orthogonal(fracture.largest_angle):
            return cls.TRANSVERSALE
        return cls.TRANSVERSALE


class Garden(Scale):
    """
    A scale used for hips fracture classification.
    See https://radiopaedia.org/articles/garden-classification-of-hip-fractures
    """

    I = 1
    II = 2
    III = 3
    IV = 4

    @property
    def stable(self) -> bool:
        return self.value() <= 2

    @property
    def treatment_suggestion(self) -> set[str]:
        return {"Internal fixation"} if self.stable else {"Arthroplasty"}


class Salter(Scale):
    """
    Classification des dégats au cartilage de conj.
    Important pour les enfants (affecte la croissance si endommagé).
    Voir https://www.nucleotype.com/salter-harris-fractures/
    """

    SLIPPED = 1
    ABOVE_PHYSIS = 2
    LOWER_PHYSIS = 3
    THROUGH_PHYSIS = 4
    ERASURE_OF_PHYSIS = 5

    @property
    def stable(self) -> bool:
        return self.value() <= 2


class Fémur(Scale):
    """
    Types de fractures du fémur
    """

    pass

    # TODO: give severity values
    # CERVICO
    # PER
    # SOUS
    # DIAPHYSAIRE


def orthogonal(angle: float, ε: float = 2.0) -> bool:
    return angle - 90 <= ε


def classify(fracture: Fracture) -> tuple[FractureType, set[Scale]]:
    return FractureType.compute(fracture), {
        scale.compute(fracture) for scale in fracture.applicable_scales
    }
