# ============================================================
# WHO AWaRe category metadata
# ============================================================
# Source:
# - WHO AWaRe classification of antibiotics for evaluation
#   and monitoring of use, 2023.
#
# Criteria:
# - Antibiotics are assigned to one of:
#   Access, Watch, Reserve.
# - If an antibiotic is not classified in the WHO AWaRe 2023
#   spreadsheet, or if AWaRe is not applicable, it is assigned None.
# - Clinical suffixes in the dataset are mapped to the base antibiotic.
#   Example:
#   Meropenem_with_meningitis -> Meropenem
#   Cefoxitin_screen -> Cefoxitin
#   Vancomycin_GRD -> Vancomycin
# - Combination therapies are classified using the specific WHO AWaRe
#   entry when available.
#   Example:
#   Ceftazidime-Avibactam -> Reserve
#   Piperacillin-Tazobactam -> Watch

antibiotic_aware_category = {
    "5-Fluorocytosine": None,
    "Amikacin": "Access",
    "Amoxicillin": "Access",
    "Amoxicillin-Clavulanic acid": "Access",
    "Amoxicillin-Clavulanic acid_uncomplicated_HWI": "Access",
    "Amphotericin B": None,
    "Ampicillin": "Access",
    "Ampicillin-Sulbactam": "Access",
    "Anidulafungin": None,
    "Azithromycin": "Watch",
    "Aztreonam": "Reserve",
    "Bacitracin": None,
    "Benzylpenicillin": "Access",
    "Benzylpenicillin_others": "Access",
    "Benzylpenicillin_with_meningitis": "Access",
    "Benzylpenicillin_with_pneumonia": "Access",
    "Caspofungin": None,
    "Cefalotin-Cefazolin": "Access",
    "Cefazolin": "Access",
    "Cefepime": "Watch",
    "Cefixime": "Watch",
    "Cefotaxime": "Watch",
    "Cefoxitin": "Watch",
    "Cefoxitin_screen": "Watch",
    "Cefpodoxime": "Watch",
    "Ceftarolin": "Reserve",
    "Ceftazidime": "Watch",
    "Ceftazidime-Avibactam": "Reserve",
    "Ceftobiprole": "Reserve",
    "Ceftolozane-Tazobactam": "Reserve",
    "Ceftriaxone": "Watch",
    "Cefuroxime": "Watch",
    "Cefuroxime.1": "Watch",
    "Chloramphenicol": "Access",
    "Ciprofloxacin": "Watch",
    "Clarithromycin": "Watch",
    "Clindamycin": "Access",
    "Clindamycin_induced": "Access",
    "Colistin": "Reserve",
    "Cotrimoxazol": "Access",
    "Cotrimoxazole": "Access",
    "Daptomycin": "Reserve",
    "Doxycycline": "Access",
    "Ertapenem": "Watch",
    "Erythromycin": "Watch",
    "Ethambutol_5mg-l": None,
    "Fluconazole": None,
    "Fosfomycin": "Watch",
    "Fusidic acid": "Watch",
    "Gentamicin": "Access",
    "Gentamicin_high_level": "Access",
    "Imipenem": "Watch",
    "Isavuconazole": None,
    "Isoniazid_.1mg-l": None,
    "Isoniazid_.4mg-l": None,
    "Itraconazole": None,
    "Levofloxacin": "Watch",
    "Linezolid": "Reserve",
    "MRSA": None,
    "Meropenem": "Watch",
    "Meropenem-Vaborbactam": "Reserve",
    "Meropenem_with_meningitis": "Watch",
    "Meropenem_with_pneumonia": "Watch",
    "Meropenem_without_meningitis": "Watch",
    "Metronidazole": "Access",
    "Micafungin": None,
    "Minocycline": "Watch",
    "Moxifloxacin": "Watch",
    "Mupirocin": None,
    "Nitrofurantoin": "Access",
    "Norfloxacin": "Watch",
    "Novobiocin": None,
    "Ofloxacin": "Watch",
    "Oxacillin": "Access",
    "Pefloxacin": "Watch",
    "Penicillin": "Access",
    "Penicillin_with_endokarditis": "Access",
    "Penicillin_with_meningitis": "Access",
    "Penicillin_with_other_infections": "Access",
    "Penicillin_with_pneumonia": "Access",
    "Penicillin_without_endokarditis": "Access",
    "Penicillin_without_meningitis": "Access",
    "Piperacillin": "Watch",
    "Piperacillin-Tazobactam": "Watch",
    "Polymyxin B": "Reserve",
    "Posaconazole": None,
    "Pristinamycin": "Watch",
    "Pyrazinamide": None,
    "Rifampicin": "Watch",
    "Rifampicin_1mg-l": "Watch",
    "Sparfloxacin": "Watch",
    "Strepomycin_high_level": "Watch",
    "Streptomycin": "Watch",
    "Teicoplanin": "Watch",
    "Teicoplanin_GRD": "Watch",
    "Telithromycin": "Watch",
    "Tetracycline": "Access",
    "Ticarcillin": "Watch",
    "Ticarcillin-Clavulan acid": "Watch",
    "Tigecycline": "Reserve",
    "Tobramycin": "Watch",
    "Vancomycin": "Watch",
    "Vancomycin_GRD": "Watch",
    "Voriconazole": None,
}