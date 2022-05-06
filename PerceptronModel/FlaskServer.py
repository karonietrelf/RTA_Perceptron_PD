from flask import Flask
from flask_restful import Resource, Api
from flask import request, jsonify
from sqlalchemy import create_engine
import numpy as np
import pickle

# Create Web application
app = Flask(__name__)
api = Api(app)

# Load model from the file
file = open('p-model.pkl', 'rb')

model = pickle.load(file)

# Create Sqlite DB Engine
db_engine = create_engine('sqlite:///iris.db')

# Create connection to create/drop table. This is persistent - comment if needed
conn = db_engine.connect()
# conn.execute('DROP TABLE results')
# conn.execute('CREATE TABLE results (id varchar(3), data json)')
conn.close()

# Global request ID
req_id = 0

# Save data to the DB as json
def save_to_db(data):
    global req_id
    req_id += 1

    conn = db_engine.connect()

    conn.execute(f"INSERT INTO results (id, data) VALUES (?, ?)", [req_id, data])
    querry = conn.execute(f'SELECT * FROM results')
    print("Querry", querry.cursor.fetchall())

    conn.close()


# Register /solve route and handler
@app.route('/solve')
def solve_task():
    sl = float(request.args.get('sl', "0"))
    pl = float(request.args.get('pl', "0"))
    print("type: ", type(sl))

    # Create data to predict
    X = np.array([sl, pl])
    print(X)
    prediction = model.predict(X)
    m_errors = model.errors_
    m_weight = model.w_

    if -1 in prediction:
        # Parse to json and save to the db
        data = {"result": f"Prediction done for [{sl}, {pl}]. Result: Versicolor",
                "errors": f"{m_errors}",
                "weight": f"{m_weight}"}
        save_to_db(jsonify(data).data)
        return f"Prediction done for [{sl}, {pl}]. Result: Versicolor"
    elif 1 in prediction:
        # Parse to json and save to the db
        data = {"result": f"Prediction done for [{sl}, {pl}]. Result: Setosa",
                "errors": f"{m_errors}",
                "weight": f"{m_weight}"}
        save_to_db(jsonify(data).data)
        return f"Prediction done for [{sl}, {pl}]. Result: Setosa"


# Implement method to get data from DB
@app.route('/get-from-db')
def get_data():
    pass


if __name__ == '__main__':
    # Press run and then...
    # Search in the web browser: localhost:5050/solve?param1=value1&param2=value2
    app.run(port=5050)
