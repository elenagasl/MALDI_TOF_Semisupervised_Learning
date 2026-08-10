# ============================================================
# Antibiotic generation metadata
# ============================================================
# Criteria:
# - Generation is encoded only for:
#   1) cephalosporins
#   2) quinolones / fluoroquinolones
#   3) tetracyclines
#
# - All other antibiotics are assigned generation = None.
# - Main source: Guía ABE, "General description of the main
#   groups of antimicrobial drugs. Antibiotics".
# - Combination therapies are classified according to the antibiotic backbone:
#   Ceftazidime-Avibactam -> ceftazidime = 3rd-generation cephalosporin
#   Ceftolozane-Tazobactam -> ceftolozane = 5th-generation cephalosporin

antibiotic_generation_metadata = {}


antibiotic_generation_metadata.update({
    # ========================================================
    # 1. CEPHALOSPORINS
    # ========================================================

    # 1st-generation cephalosporins
    "Cefalotin-Cefazolin": {
        "generation_family": "cephalosporin",
        "generation": 1,
    },
    "Cefazolin": {
        "generation_family": "cephalosporin",
        "generation": 1,
    },

    # 2nd-generation cephalosporins
    "Cefoxitin": {
        "generation_family": "cephalosporin",
        "generation": 2,
    },
    "Cefoxitin_screen": {
        "generation_family": "cephalosporin",
        "generation": 2,
    },
    "Cefuroxime": {
        "generation_family": "cephalosporin",
        "generation": 2,
    },
    "Cefuroxime.1": {
        "generation_family": "cephalosporin",
        "generation": 2,
    },

    # 3rd-generation cephalosporins
    "Cefixime": {
        "generation_family": "cephalosporin",
        "generation": 3,
    },
    "Cefotaxime": {
        "generation_family": "cephalosporin",
        "generation": 3,
    },
    "Cefpodoxime": {
        "generation_family": "cephalosporin",
        "generation": 3,
    },
    "Ceftazidime": {
        "generation_family": "cephalosporin",
        "generation": 3,
    },
    "Ceftazidime-Avibactam": {
        "generation_family": "cephalosporin",
        "generation": 3,
    },
    "Ceftriaxone": {
        "generation_family": "cephalosporin",
        "generation": 3,
    },

    # 4th-generation cephalosporins
    "Cefepime": {
        "generation_family": "cephalosporin",
        "generation": 4,
    },

    # 5th-generation cephalosporins
    "Ceftarolin": {
        "generation_family": "cephalosporin",
        "generation": 5,
    },
    "Ceftobiprole": {
        "generation_family": "cephalosporin",
        "generation": 5,
    },
    "Ceftolozane-Tazobactam": {
        "generation_family": "cephalosporin",
        "generation": 5,
    },


    # ========================================================
    # 2. QUINOLONES / FLUOROQUINOLONES
    # ========================================================

    # 2nd-generation quinolones / fluoroquinolones
    "Ciprofloxacin": {
        "generation_family": "quinolone",
        "generation": 2,
    },
    "Norfloxacin": {
        "generation_family": "quinolone",
        "generation": 2,
    },
    "Ofloxacin": {
        "generation_family": "quinolone",
        "generation": 2,
    },
    "Pefloxacin": {
        "generation_family": "quinolone",
        "generation": 2,
    },

    # 3rd-generation quinolones / fluoroquinolones
    "Levofloxacin": {
        "generation_family": "quinolone",
        "generation": 3,
    },
    "Sparfloxacin": {
        "generation_family": "quinolone",
        "generation": 3,
    },

    # 4th-generation quinolones / fluoroquinolones
    "Moxifloxacin": {
        "generation_family": "quinolone",
        "generation": 4,
    },


    # ========================================================
    # 3. TETRACYCLINES
    # ========================================================

    # 1st-generation tetracyclines
    "Tetracycline": {
        "generation_family": "tetracycline",
        "generation": 1,
    },

    # 2nd-generation tetracyclines
    "Doxycycline": {
        "generation_family": "tetracycline",
        "generation": 2,
    },
    "Minocycline": {
        "generation_family": "tetracycline",
        "generation": 2,
    },

    # 3rd-generation tetracyclines
    "Tigecycline": {
        "generation_family": "tetracycline",
        "generation": 3,
    },
})
