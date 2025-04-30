from dataclasses import dataclass


__all__ = ["EvalQuestion", "get_eval_questions"]


@dataclass
class EvalQuestion:
    id: int
    text: str


def get_eval_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(idx, text) for idx, text in enumerate(QUESTIONS_STR)
    ]


QUESTIONS_STR = [
    """
Are AWI Quality Certification Program (QCP) Labels Required?
i. If AWI requirements are stated, do they truly need the AWI QCP Labels, or will our AWI Certification suffice?
    """,

    """
Are there any LEED Requirements for the project?  
i. FSC?
ii. NAUF?
iii. Regionally Sourced Materials?
    """,

    """
Approved Manufacturers – Is Stevens listed as an Approved Manufacturer or is a Substitution Request needed for this?
    """,

    """
What Core are the Cabinets required to be made of?
i. Particleboard, MDF, or Plywood?
ii. Are NAUF Materials required?
    """,

    """
What Finish Material Requirements are there for the Front Surfaces, Exposed vs. Semi-Exposed Surfaces, Interiors, etc.?
i. HPL, TFL/Melamine, Wood Veneers?
    """,

    """
What grain direction is required?
i. Drawer grain same as door grain?
ii. Drawer grain opposite door grain?
iii. Vertical Grain Matching required?
    """,

    """
What Edgings are Required for the project?
i. Door and Drawer Fronts vs. Cabinet Bodies?                                                            
ii. 0.5mm, 1mm, 3mm ?
    """,

    """
Overall Question for all Hardware Items…..Are there any specialty Hardware Brand requirements, or will Steven’s standards or BHMA standards be approved?
i. Is there any qualifying statement from the Architect showing that any hardware manufacturer is acceptable as long as the product meets Form, Fit, and Function?
    """,

    """
What type of hinges are required?
i. Five Knuckle
ii. Concealed (120° vs. 165°)
iii. Specialty Hinge Requirements (Piano Hinges, Inset Hinges, etc.)
    """,

    """
What type of drawer slides will be required?
 i. Full Extension Ball Bearing
 ii. ¾ Extension Epoxy Coated
 iii. Will Stevens 100lb Full Extension Slides be acceptable on File Drawers or will 150lb-200lb File Drawer Slides be required?
 iv. Full extension slides required on file drawers only, or all drawers?
    """,

    """
What type of pulls are required?
 i. Length (4” vs. 5”)
 ii. Standard Bentwire
 iii. Specialty Pulls by preselected Manufacturers?
 iv. Tab Pulls, Inset Pulls, etc?
    """,

    """
What type of door catch will be required?
 i. Magnetic
 ii. Roller
    """,

    """
What locks will be required?
i. SII Standard Approved or BHMA Number Referencing E07121 & E07122 standards?
ii. Best Lock Requirements?
If so, Construction Cores provided only.
iii. Lock on all or as indicated on drawings or lock schedule?
    """,

    """
Shelf Supports
i. BHMA B04013 (SII Standard ) vs. B04061/B04071 (K&V Recessed Shelving Standards)
ii. Are SII Polycarbonate Twin-Pin Anti-Tip shelf rests acceptable?
    """,

    """
What type of toekicks are required?
i. Integral
ii. Separate Base (If so, what Materials?)
    """,

    """
What type of cabinet backs are required?
 i. 1/4” vs. ½” Backs
 ii. Particle board or MDF
 iii. Melamine or Pre-finished
 iv. Will Stevens standard pre-finished backs be acceptable?
    """,

    """
What drawer box construction method is required?
i. Thermofused Laminate vs. Hardwood
ii. Dowel Pin Construction vs. Dovetail Drawer
    """,

    """
What material will be required for adjustable shelves?
i. Particleboard
ii. Plywood
    """,

    """
What adjustable shelving thickness will be required?
i. ¾” thick up to 36” wide, 1” over 36” wide
ii. 1” all adjustable shelves
iii. ¾” thick up to 36” wide, 1” over 36” wide and 1” on all open
    """
]
