class MOF:
    uuid = ''
    filename = ''
    LCD = 0
    PLD = 0
    LFPD = 0
    cm3_g = 0
    ASA_m2_cm3 = 0
    ASA_m2_g = 0
    AV_VF = 0
    AV_cm3_g = 0

    def __init__(self, uuid, filename, LCD, PLD, LFPD, cm3_g, ASA_m2_cm3, ASA_m2_g, AV_VF, AV_cm3_g):
        self.uuid = uuid
        self.filename = filename
        self.LCD = LCD
        self.PLD = PLD
        self.LFPD = LFPD
        self.cm3_g = cm3_g
        self.ASA_m2_cm3 = ASA_m2_cm3
        self.ASA_m2_g = ASA_m2_g
        self.AV_VF = AV_VF
        self.AV_cm3_g = AV_cm3_g