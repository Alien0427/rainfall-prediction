import sqlite3
from datetime import datetime
import json

class Database:
    def __init__(self, db_name='rainfall_predictions.db'):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        # Create predictions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                pressure REAL,
                prediction BOOLEAN,
                probability REAL,
                location TEXT
            )
        ''')
        
        # Create locations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                latitude REAL,
                longitude REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_prediction(self, data):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO predictions 
            (timestamp, temperature, humidity, wind_speed, pressure, prediction, probability, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            data['temperature'],
            data['humidity'],
            data['wind_speed'],
            data['pressure'],
            data['prediction'],
            data['probability'],
            data.get('location', 'Unknown')
        ))
        
        conn.commit()
        conn.close()

    def get_recent_predictions(self, limit=10):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        predictions = c.fetchall()
        conn.close()
        
        return [{
            'id': p[0],
            'timestamp': p[1],
            'temperature': p[2],
            'humidity': p[3],
            'wind_speed': p[4],
            'pressure': p[5],
            'prediction': p[6],
            'probability': p[7],
            'location': p[8]
        } for p in predictions]

    def add_location(self, name, latitude, longitude):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO locations (name, latitude, longitude)
                VALUES (?, ?, ?)
            ''', (name, latitude, longitude))
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            success = False
        
        conn.close()
        return success

    def get_locations(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('SELECT * FROM locations')
        locations = c.fetchall()
        conn.close()
        
        return [{
            'id': l[0],
            'name': l[1],
            'latitude': l[2],
            'longitude': l[3]
        } for l in locations] 