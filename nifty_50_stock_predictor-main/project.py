from flask import Flask , render_template , jsonify, request,send_file
from flask_sqlalchemy  import SQLAlchemy 
from datetime import datetime
from _StockPrice import stockPredictionModel

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///project.db"
db = SQLAlchemy (app)



class User(db.Model):
    S_no = db.Column(db.Integer, primary_key=True)
    number_stock = db.Column(db.Integer, primary_key=False , nullable=False)
    start_date = db.Column(db.String(80), unique=True, nullable=False)
    end_date = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f" {self.username} - {self.number_stock} - {self.start_date} - {self.end_date}"

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/test', methods=['POST']) 
def foo():
    # data = request.json
    # response = jsonify(data)
    # print(response)
    request_json = request.get_json()
    stockName = request_json.get('stockName')
    stockNumber = request_json.get('stockNumber')
    startDate = request_json.get('startDate')
    endDate = request_json.get('endDate')
    print(stockName, stockNumber, startDate, endDate)
    response2 = stockPredictionModel(stockName, stockNumber, startDate, endDate)
    print(response2)
    return { 'result': response2}
    




if __name__ == '__main__':
    app.run(debug = True)
    