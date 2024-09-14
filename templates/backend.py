from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# Setup MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.introspectacale
conversations = db.conversations

@app.route('/save-conversation', methods=['POST'])
def save_conversation():
    data = request.json
    name = data.get('name')
    notes = data.get('notes')

    if name and notes:
        conversation = {
            'name': name,
            'notes': notes
            # You can add other fields like date, time, or recorded video if needed
        }
        conversations.insert_one(conversation)
        return jsonify({'message': 'Conversation saved successfully!'}), 201
    else:
        return jsonify({'error': 'Missing data'}), 400

if __name__ == '__main__':
    app.run(debug=True)