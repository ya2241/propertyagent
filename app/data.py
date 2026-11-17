"""Synthetic lettings inventory, viewing slots and an in-memory booking store."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class Property:
    id: str
    title: str
    area: str
    rent_pcm: int
    bedrooms: int
    available_from: date
    description: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.area}. {self.description}"


@dataclass(frozen=True)
class Slot:
    slot_id: str
    property_id: str
    start: datetime


@dataclass(frozen=True)
class Booking:
    booking_id: str
    property_id: str
    slot_id: str
    applicant_name: str
    applicant_email: str


_RAW = [
    ("P001", "Sunny 1-bed flat overlooking the park", "Hackney", 1850, 1, "2026-12-01",
     "Bright first-floor flat overlooking a leafy park. Quiet street, double glazing, ideal for a professional working from home with a desk nook. Ten minutes' walk to the overground station."),
    ("P002", "Pet-friendly 2-bed house with garden", "Walthamstow", 2300, 2, "2026-12-15",
     "Terraced house with private garden and a cat flap. Dogs and cats welcome with a pet deposit. Close to the marshes for dog walking and a family-friendly local primary school."),
    ("P003", "Compact studio near King's Cross", "King's Cross", 1450, 0, "2026-11-25",
     "Compact modern studio with kitchenette, great for a single professional commuting by train. Excellent transport links, gym in the building, concierge."),
    ("P004", "Spacious 3-bed family home", "Croydon", 2100, 3, "2027-01-10",
     "Semi-detached with three bedrooms, large kitchen-diner and off-street parking. Near good schools and parks, suited to families with children. Garden with shed."),
    ("P005", "Riverside 2-bed apartment", "Greenwich", 2600, 2, "2027-01-05",
     "Modern apartment with balcony facing the Thames, secure underground parking, lift and 24-hour concierge. Spare room works as a home office."),
    ("P006", "Budget 1-bed above the high street", "Lewisham", 1300, 1, "2026-12-01",
     "Affordable one bedroom flat above shops, lively high street, fast trains to London Bridge. Bills not included. Suitable for a student or young professional on a tight budget."),
    ("P007", "Quiet 2-bed with study", "Islington", 2750, 2, "2027-02-01",
     "Peaceful top-floor flat with a small second bedroom used as a study, ideal for remote work. Character building, original floorboards, tree-lined street."),
    ("P008", "Large 4-bed house for sharers", "Brixton", 3200, 4, "2026-12-20",
     "Spacious house suitable for sharers or a group of friends, two bathrooms, big living room, short walk to the tube. Bright and sociable."),
    ("P009", "Cosy 1-bed garden flat", "Clapham", 1950, 1, "2026-12-10",
     "Ground-floor garden flat with own entrance and patio. Small pets considered. Quiet residential road close to the common."),
    ("P010", "Modern 2-bed new build", "Stratford", 2400, 2, "2027-01-15",
     "New-build apartment with gym, communal roof terrace and fast broadband. Next to the Elizabeth line and shopping centre, popular with young professionals."),
    ("P011", "Period 3-bed maisonette", "Highbury", 3400, 3, "2027-03-01",
     "Elegant maisonette spread over two floors with high ceilings, open fireplace and a private terrace. Close to cafes and the park."),
    ("P012", "Furnished studio in the heart of the action", "Shoreditch", 1700, 0, "2026-12-05",
     "Stylish furnished studio in the nightlife area. Lively and noisy at weekends, walkable to bars and restaurants, fast fibre internet."),
    ("P013", "Family 3-bed with driveway", "Bromley", 2200, 3, "2026-12-18",
     "Detached house with driveway parking, big garden, quiet cul-de-sac. Pets allowed, great for families with children and dogs, excellent schools nearby."),
    ("P014", "High-floor 1-bed with skyline views", "Canary Wharf", 2050, 1, "2026-11-30",
     "High-floor flat with skyline views, residents' gym and pool, short commute to the financial district, concierge, secure bike storage."),
    ("P015", "Light 2-bed near the market", "Peckham", 2000, 2, "2026-12-12",
     "Airy flat with large windows, close to independent cafes, arts venues and a vibrant market. Cycle friendly with bike storage."),
    ("P016", "Affordable studio by the Victoria line", "Walthamstow", 1250, 0, "2026-12-08",
     "Small tidy studio next to the Victoria line. Low cost option for students or a single person on a tight budget. Bills included."),
    ("P017", "Quiet 2-bed bungalow", "Enfield", 1900, 2, "2027-01-20",
     "Single-storey home on a calm street, step-free access suitable for wheelchair users or older tenants, garden, driveway parking. Pets considered."),
    ("P018", "Luxury 3-bed penthouse", "Battersea", 5200, 3, "2027-02-10",
     "Top-floor penthouse with panoramic terrace, concierge, spa and underground parking. Premium finishes throughout, suited to executives."),
    ("P019", "2-bed flat near the university", "Camden", 2850, 2, "2026-12-22",
     "Walking distance to the university campus, libraries and museums. Furnished, suitable for postgraduate students or academics sharing."),
    ("P020", "Eco-friendly 1-bed", "Dalston", 1900, 1, "2027-01-08",
     "Energy-efficient flat with solar panels, low energy bills, bike storage and a communal garden. Quiet courtyard, suited to remote workers who value sustainability."),
]

PROPERTIES = [
    Property(i, t, a, r, b, date.fromisoformat(d), desc) for i, t, a, r, b, d, desc in _RAW
]


class PropertyStore:
    def __init__(self, properties: list[Property] | None = None):
        self.properties = {p.id: p for p in (properties or PROPERTIES)}
        base = datetime(2026, 11, 23)
        times = [(10, 0), (14, 0), (17, 30)]
        self._slots: dict[str, list[Slot]] = {}
        for i, pid in enumerate(self.properties):
            self._slots[pid] = [
                Slot(f"{pid}-S{n + 1}", pid, (base + timedelta(days=(i + n) % 10)).replace(hour=h, minute=m))
                for n, (h, m) in enumerate(times)
            ]
        self.bookings: list[Booking] = []

    def get(self, property_id: str) -> Property | None:
        return self.properties.get(property_id)

    def free_slots(self, property_id: str) -> list[Slot]:
        taken = {b.slot_id for b in self.bookings}
        return [s for s in self._slots.get(property_id, []) if s.slot_id not in taken]

    def slot(self, slot_id: str) -> Slot | None:
        return next((s for slots in self._slots.values() for s in slots if s.slot_id == slot_id), None)
