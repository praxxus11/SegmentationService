import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

class DB:
    def __init__(self, sqlite_db_path):
        self.con = sqlite3.connect(sqlite_db_path)
        self.cur = self.con.cursor()

        self.cur.execute('''
        CREATE TABLE IF NOT EXISTS Meta (
            img_id TEXT PRIMARY KEY,
            start_mili INTEGER,
            end_mili INTEGER,
            num_masks INTEGER,
            num_threads INTEGER
        )
        ''')
        self.cur.execute('''
        CREATE TABLE IF NOT EXISTS ClassificationMeta (
            img_id TEXT,
            pitcher_id TEXT,
            pred_species_1 TEXT,
            pred_species_1_conf REAL,
            pred_species_2 TEXT,
            pred_species_2_conf REAL,
            start_mili INTEGER,
            end_mili INTEGER,
            num_threads INTEGER,
            PRIMARY KEY (img_id, pitcher_id),
            FOREIGN KEY (img_id) REFERENCES Meta (img_id)
        )
        ''')

        self.con.commit()
    
    def insert_into_meta(self, meta):
        insert_sql = '''
        INSERT INTO Meta (img_id, start_mili, end_mili, num_masks, num_threads)
        VALUES (?, ?, ?, ?, ?)
        '''
        self.cur.execute(insert_sql, (
            meta.img_id,
            meta.start_mili,
            meta.end_mili,
            meta.num_masks,
            meta.num_threads,
        ))
        self.con.commit()
    
    def insert_into_class_meta(self, img_id, classification_meta):
        insert_sql = '''
        INSERT INTO ClassificationMeta (img_id, pitcher_id, pred_species_1, pred_species_1_conf, pred_species_2, pred_species_2_conf, start_mili, end_mili, num_threads)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cur.execute(insert_sql, (
            img_id,
            classification_meta.pitcher_id,
            classification_meta.pred_species_1,
            classification_meta.pred_species_1_conf,
            classification_meta.pred_species_2,
            classification_meta.pred_species_2_conf,
            classification_meta.start_mili,
            classification_meta.end_mili,
            classification_meta.num_threads,
        ))
        self.con.commit()

db = DB(os.path.join(os.environ['DB_DIR'], 'inference.db'))

def dump_meta(meta):
    db.insert_into_meta(meta)
    for class_meta in meta.classifications:
        db.insert_into_class_meta(meta.img_id, class_meta)
    logger.info(f"Dumped meta for {meta.img_id}")
