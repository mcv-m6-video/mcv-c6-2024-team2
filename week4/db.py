import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import sqlite3

class DatabaseHandler:
    def __init__(self, database_name='feature_database.db'):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS features
                            (track_id INTEGER, frame_id INTEGER, seq TEXT, cam TEXT, feature_vector BLOB, reid BOOLEAN)''')
        self.conn.commit()

    def insert_feature(self, track_id, frame_id, seq, cam, feature_vector, reid):
        feature_vector_bytes = feature_vector.tobytes()
        self.cursor.execute("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)",
                            (track_id, frame_id, seq, cam, feature_vector_bytes, reid))
        self.conn.commit()

    def is_track_id_exists(self, track_id, seq):
        self.cursor.execute("SELECT * FROM features WHERE track_id=? AND seq=?", (track_id,seq))
        result = self.cursor.fetchone()
        return result is not None
    
    def get_object(self, track_id, seq):
        self.cursor.execute("SELECT * FROM features WHERE track_id=? AND seq=?", (track_id,seq))
        result = self.cursor.fetchone()
        return result

    def get_between_frames(self, min_frame_id, max_frame_id):
        self.cursor.execute("SELECT * FROM features WHERE frame_id BETWEEN ? AND ?", (min_frame_id, max_frame_id))
        entries = self.cursor.fetchall()
        return entries
    
    def update_reid(self, track_id, seq, new_reid):
        self.cursor.execute("UPDATE features SET reid=? WHERE track_id=? AND seq=?", (new_reid, track_id, seq))
        self.conn.commit()

    def get_between_frames_not_in_camera(self, min_frame_id, max_frame_id, cam):
        self.cursor.execute("SELECT * FROM features WHERE frame_id BETWEEN ? AND ? camera != ?", (min_frame_id, max_frame_id, cam))
        entries = self.cursor.fetchall()
        return entries

    def close_connection(self):
        self.conn.close()

