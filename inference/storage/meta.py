class Meta:
    def __init__(self):
        self.img_id = None
        self.start_mili = None
        self.end_mili = None
        self.num_masks = None
        self.classifications = []
        self.num_threads = None

class ClassificationMeta:
    def __init__(self):
        self.pitcher_id = None
        self.pred_species_1 = None
        self.pred_species_1_conf = None
        self.pred_species_2 = None
        self.pred_sepcies_2_conf = None
        self.start_mili = None
        self.end_mili = None
        self.num_threads = None
